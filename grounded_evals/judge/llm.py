"""LLM judge speaking to OpenAI-compatible or native Anthropic endpoints.

Transport: standard-library HTTP with bounded exponential backoff on 429/5xx
and network errors. Anthropic's Messages API is detected from the base URL;
everything else is treated as OpenAI ``chat/completions``.

Structured output is parsed *and validated* (required keys, types, ranges,
span-index bounds). A judge that returns malformed output is re-asked up to
``max_retries`` times with the error appended. Terminal failures diverge by
call, on purpose:

- ``extract_claims`` raises ``JudgeError`` — with no claims there is nothing
  to route, and a silent empty list would read as a perfectly grounded answer.
- ``verdict`` returns the failure sentinel (UNSUPPORTED, confidence 0.0),
  which ``ClaimResult.needs_review`` always routes to human review.

Configuration is explicit: pass ``model``/``api_key``/``base_url`` to the
constructor. ``LLMJudge.from_env()`` is the only place environment variables
are read (GROUNDED_EVALS_MODEL, GROUNDED_EVALS_API_KEY, GROUNDED_EVALS_BASE_URL).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from ..schema import Citation, Claim, JudgeVerdict, SourceSpan, Verdict
from .base import EXTRACT_PROMPT, PROMPT_VERSION, VERDICT_PROMPT, JudgeError

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"

_VERDICT_VALUES = frozenset(
    v.value for v in (Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.CONTRADICTED, Verdict.UNSUPPORTED)
)
_HTTP_ATTEMPTS = 3
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMJudge:
    """Judge implementation backed by a hosted chat model."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_output_tokens: int = 1024,
    ):
        if not model or not api_key:
            raise ValueError("LLMJudge requires a model and an api_key")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_env(cls) -> LLMJudge:
        """Build from GROUNDED_EVALS_* environment variables.

        The only environment read in this module — library code takes
        explicit arguments.

        Raises:
            ValueError: If the required variables are unset (configuration
                errors are ``ValueError``, like the constructor's;
                ``JudgeError`` is reserved for runtime judge failures).
        """
        model = os.environ.get("GROUNDED_EVALS_MODEL", "")
        api_key = os.environ.get("GROUNDED_EVALS_API_KEY", "")
        base_url = os.environ.get("GROUNDED_EVALS_BASE_URL", OPENAI_DEFAULT_BASE_URL)
        if not model or not api_key:
            raise ValueError(
                "LLMJudge.from_env needs GROUNDED_EVALS_MODEL and "
                "GROUNDED_EVALS_API_KEY (use MockJudge for keyless runs)"
            )
        return cls(model=model, api_key=api_key, base_url=base_url)

    @property
    def fingerprint(self) -> dict:
        """Judge identity for reproducibility records in reports."""
        return {
            "judge": "llm",
            "model": self.model,
            "base_url": self.base_url,
            "prompt_version": PROMPT_VERSION,
        }

    # -- transport ---------------------------------------------------------
    def _is_anthropic(self) -> bool:
        return "anthropic" in urllib.parse.urlparse(self.base_url).netloc

    def _build_request(self, prompt: str) -> tuple[str, bytes, dict[str, str]]:
        """Pure request builder (separately testable, no network)."""
        if self._is_anthropic():
            url = f"{self.base_url}/messages"
            body = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            }
        else:
            url = f"{self.base_url}/chat/completions"
            body = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        return url, json.dumps(body).encode(), headers

    def _parse_response(self, payload: dict) -> str:
        """Pure response text extractor for both API shapes."""
        if self._is_anthropic():
            return "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
        return payload["choices"][0]["message"]["content"]

    def _chat(self, prompt: str) -> str:
        url, body, headers = self._build_request(prompt)
        last_err: Exception | None = None
        for attempt in range(_HTTP_ATTEMPTS):
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return self._parse_response(json.load(r))
            except urllib.error.HTTPError as e:
                if e.code not in _RETRYABLE_STATUS:
                    raise JudgeError(f"judge endpoint returned HTTP {e.code}") from e
                last_err = e
            except urllib.error.URLError as e:
                last_err = e
            if attempt < _HTTP_ATTEMPTS - 1:
                time.sleep(2**attempt)
        raise JudgeError(f"judge endpoint unreachable after {_HTTP_ATTEMPTS} attempts: {last_err}")

    # -- structured output -------------------------------------------------
    @staticmethod
    def _json(text: str) -> dict:
        """Extract the first JSON *object* from the reply.

        Scans brace-balanced candidates with ``raw_decode`` so prose
        containing stray braces around the object does not defeat parsing,
        and rejects non-object JSON ("[]", "42", "null") — every judge schema
        here is an object, and a non-dict must surface as a validation error,
        never as an ``AttributeError`` deeper in the pipeline.
        """
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        idx = text.find("{")
        while idx != -1:
            try:
                candidate, _ = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
                continue
            if isinstance(candidate, dict):
                return candidate
            idx = text.find("{", idx + 1)
        raise ValueError("no JSON object in judge output")

    def _ask(self, prompt: str, validate: Any) -> dict:
        """Ask, parse, validate; re-ask with the error message on failure."""
        err = ""
        for _ in range(self.max_retries + 1):
            suffix = (
                f"\n\nYour previous output was invalid: {err}. Return only valid JSON."
                if err
                else ""
            )
            raw = self._chat(prompt + suffix)
            try:
                data = self._json(raw)
                validate(data)
                return data
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
        raise ValueError(f"judge output invalid after retries: {err}")

    @staticmethod
    def _validate_extract(data: dict) -> None:
        claims = data.get("claims")
        if not isinstance(claims, list):
            raise ValueError('"claims" must be a list')
        for c in claims:
            if not isinstance(c, dict):
                raise ValueError("each claim must be an object")
            if not isinstance(c.get("text"), str) or not c["text"].strip():
                raise ValueError('each claim needs a non-empty "text" string')
            cites = c.get("citations", [])
            if not isinstance(cites, list) or not all(
                isinstance(x, (str, int)) and not isinstance(x, bool) for x in cites
            ):
                raise ValueError('"citations" must be a list of marker strings')
            if "checkable" in c and not isinstance(c["checkable"], bool):
                raise ValueError('"checkable" must be a boolean')

    @staticmethod
    def _validate_verdict(data: dict, n_spans: int) -> None:
        if data.get("verdict") not in _VERDICT_VALUES:
            raise ValueError(f'"verdict" must be one of {sorted(_VERDICT_VALUES)}')
        conf = data.get("confidence")
        if (
            isinstance(conf, bool)  # bool passes isinstance(int) — reject it
            or not isinstance(conf, (int, float))
            or not 0.0 <= float(conf) <= 1.0
        ):
            raise ValueError('"confidence" must be a number in [0, 1]')
        supporting = data.get("supporting_spans", [])
        if not isinstance(supporting, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n_spans
            for i in supporting
        ):
            raise ValueError(f'"supporting_spans" must be a list of span indices in [0, {n_spans})')

    # -- protocol ----------------------------------------------------------
    def extract_claims(self, question: str, answer: str) -> list[Claim]:
        prompt = EXTRACT_PROMPT.format(question=question, answer=answer)
        try:
            data = self._ask(prompt, self._validate_extract)
        except ValueError as e:
            raise JudgeError(f"claim extraction failed: {e}") from e
        return [
            Claim(
                claim_id=f"c{i}",
                text=c["text"].strip(),
                citations=[Citation(doc_id=str(x)) for x in c.get("citations", [])],
                checkable=bool(c.get("checkable", True)),
            )
            for i, c in enumerate(data["claims"])
        ]

    def verdict(self, claim: Claim, spans: Sequence[SourceSpan]) -> JudgeVerdict:
        span_text = (
            "\n".join(f"[{i}] ({s.doc_id}) {s.text}" for i, s in enumerate(spans)) or "(none)"
        )
        prompt = VERDICT_PROMPT.format(claim=claim.text, spans=span_text)
        try:
            data = self._ask(prompt, lambda d: self._validate_verdict(d, len(spans)))
        except (ValueError, JudgeError) as e:
            # The failure sentinel: confidence 0.0 always routes to review.
            return JudgeVerdict(Verdict.UNSUPPORTED, 0.0, f"judge failure -> review: {e}")
        return JudgeVerdict(
            Verdict(data["verdict"]),
            float(data["confidence"]),
            str(data.get("rationale", "")),
            tuple(data.get("supporting_spans", [])),
        )
