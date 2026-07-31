"""LLM judge speaking to any OpenAI- or Anthropic-compatible endpoint.

Structured output is validated and retried: a judge that returns malformed
JSON is re-asked up to MAX_RETRIES times with the parse error appended, then
the claim is routed to review (confidence 0.0) rather than silently dropped.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from ..schema import Citation, Claim, SourceSpan, Verdict
from .base import EXTRACT_PROMPT, VERDICT_PROMPT

MAX_RETRIES = 2


class LLMJudge:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("GROUNDED_EVALS_MODEL", "")
        self.api_key = api_key or os.environ.get("GROUNDED_EVALS_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "GROUNDED_EVALS_BASE_URL", "https://api.openai.com/v1"
        )
        if not (self.model and self.api_key):
            raise RuntimeError(
                "LLMJudge needs GROUNDED_EVALS_MODEL and GROUNDED_EVALS_API_KEY "
                "(use MockJudge for keyless runs)"
            )

    # -- transport ---------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        return out["choices"][0]["message"]["content"]

    @staticmethod
    def _json(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object in judge output")
        return json.loads(m.group())

    def _ask(self, prompt: str) -> dict:
        err = ""
        for _ in range(MAX_RETRIES + 1):
            raw = self._chat(prompt + (f"\n\nYour previous output failed to parse: {err}. Return only valid JSON." if err else ""))
            try:
                return self._json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
        raise ValueError(f"judge output unparseable after retries: {err}")

    # -- protocol ----------------------------------------------------------
    def extract_claims(self, question: str, answer: str) -> list[Claim]:
        data = self._ask(EXTRACT_PROMPT.format(question=question, answer=answer))
        claims = []
        for i, c in enumerate(data.get("claims", [])):
            claims.append(
                Claim(
                    claim_id=f"c{i}",
                    text=str(c["text"]),
                    citations=[Citation(doc_id=str(x)) for x in c.get("citation_ids", [])],
                    checkable=bool(c.get("checkable", True)),
                )
            )
        return claims

    def verdict(
        self, claim: Claim, spans: list[SourceSpan]
    ) -> tuple[Verdict, float, str]:
        span_text = "\n".join(f"- ({s.doc_id}) {s.text}" for s in spans) or "(none)"
        try:
            data = self._ask(VERDICT_PROMPT.format(claim=claim.text, spans=span_text))
            return (
                Verdict(data["verdict"]),
                max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                str(data.get("rationale", "")),
            )
        except (ValueError, KeyError) as e:
            # Unparseable/invalid verdicts route to review, never silently pass.
            return Verdict.UNSUPPORTED, 0.0, f"judge failure -> review: {e}"

