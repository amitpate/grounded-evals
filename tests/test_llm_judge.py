"""LLMJudge under test with a stubbed transport — no keys, no network.

The bug class these tests exist for: an unrenderable prompt template or an
unvalidated response path silently disabling the whole LLM judge. CI must
fail loudly on either.
"""
import json

import pytest

from grounded_evals import Citation, Claim, SourceSpan, Verdict, evaluate
from grounded_evals.judge.base import EXTRACT_PROMPT, VERDICT_PROMPT, JudgeError
from grounded_evals.judge.llm import LLMJudge
from grounded_evals.schema import SourceDoc


def _span(start: int, text: str) -> SourceSpan:
    return SourceSpan("d1", start, start + len(text), text)


SPANS = [
    _span(0, "The stadium seats 90,000 people."),
    _span(33, "It reopened in 2007."),
]


class StubJudge(LLMJudge):
    """LLMJudge with the network replaced by a scripted list of replies."""

    def __init__(self, replies):
        super().__init__(model="stub-model", api_key="stub-key")
        self.replies = list(replies)
        self.prompts = []

    def _chat(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_prompt_templates_render():
    # str.format on templates with literal JSON braces once raised KeyError
    # and disabled the entire LLM path. Never again.
    rendered = EXTRACT_PROMPT.format(question="q?", answer="a.")
    assert '{"claims"' in rendered and "<answer>" in rendered
    rendered = VERDICT_PROMPT.format(claim="c", spans="[0] s")
    assert '{"verdict"' in rendered and "<claim>" in rendered


def test_extract_claims_parses_and_preserves_markers():
    judge = StubJudge(
        [
            json.dumps(
                {
                    "claims": [
                        {"text": "Wembley seats 90,000.", "citations": ["1"], "checkable": True},
                        {"text": "It is iconic.", "citations": [], "checkable": False},
                    ]
                }
            )
        ]
    )
    claims = judge.extract_claims("q", "a")
    assert [c.text for c in claims] == ["Wembley seats 90,000.", "It is iconic."]
    assert claims[0].citations == [Citation(doc_id="1")]
    assert not claims[1].checkable


def test_extract_claims_retries_then_raises_judge_error():
    judge = StubJudge(["not json", "still not json", "{}"])  # {} fails validation too
    with pytest.raises(JudgeError):
        judge.extract_claims("q", "a")
    assert len(judge.prompts) == 3  # initial + max_retries
    assert "invalid" in judge.prompts[1]  # error fed back to the model


def test_verdict_happy_path():
    judge = StubJudge(
        [
            json.dumps(
                {
                    "verdict": "supported",
                    "supporting_spans": [0, 1],
                    "confidence": 0.9,
                    "rationale": "both spans entail it",
                }
            )
        ]
    )
    jv = judge.verdict(Claim("c0", "Seats 90,000 and reopened in 2007."), SPANS)
    assert jv.verdict == Verdict.SUPPORTED
    assert jv.supporting == (0, 1)
    assert jv.confidence == 0.9


def test_verdict_recovers_after_one_invalid_reply():
    judge = StubJudge(
        [
            json.dumps({"verdict": "definitely", "confidence": 0.9}),  # bad enum
            json.dumps(
                {"verdict": "partial", "supporting_spans": [1], "confidence": 0.6, "rationale": "r"}
            ),
        ]
    )
    jv = judge.verdict(Claim("c0", "x"), SPANS)
    assert jv.verdict == Verdict.PARTIAL


def test_verdict_terminal_failure_returns_review_sentinel():
    judge = StubJudge(["nope", "nope", "nope"])
    jv = judge.verdict(Claim("c0", "x"), SPANS)
    assert jv.verdict == Verdict.UNSUPPORTED
    assert jv.confidence == 0.0  # the sentinel ClaimResult.needs_review catches


def test_verdict_rejects_out_of_range_span_indices():
    judge = StubJudge(
        [
            json.dumps({"verdict": "supported", "supporting_spans": [7], "confidence": 0.9}),
            json.dumps({"verdict": "supported", "supporting_spans": [0], "confidence": 0.9}),
        ]
    )
    jv = judge.verdict(Claim("c0", "x"), SPANS)
    assert jv.supporting == (0,)


def test_end_to_end_with_stubbed_llm_routes_failures_to_review():
    doc = SourceDoc("d1", "t", "The stadium seats 90,000 people. It reopened in 2007.")
    judge = StubJudge(
        [
            json.dumps(
                {"claims": [{"text": "The stadium reopened in 2007.", "citations": ["1"]}]}
            ),
            "garbage",
            "garbage",
            "garbage",  # verdict fails terminally
        ]
    )
    report = evaluate("q", "ignored", [doc], judge)
    assert report.results[0].confidence == 0.0
    assert report.results[0] in report.review_queue  # failure never silently passes


def test_request_builder_openai_and_anthropic_shapes():
    openai = LLMJudge(model="m", api_key="k")
    url, body, headers = openai._build_request("hi")
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer k"
    assert json.loads(body)["temperature"] == 0

    anthropic = LLMJudge(model="m", api_key="k", base_url="https://api.anthropic.com/v1")
    url, body, headers = anthropic._build_request("hi")
    assert url.endswith("/messages")
    assert headers["x-api-key"] == "k"
    assert json.loads(body)["max_tokens"] > 0

    payload = {"content": [{"type": "text", "text": '{"a": 1}'}]}
    assert anthropic._parse_response(payload) == '{"a": 1}'


def test_constructor_requires_explicit_config():
    with pytest.raises(ValueError):
        LLMJudge(model="", api_key="k")
    with pytest.raises(ValueError):
        LLMJudge(model="m", api_key="")


def test_non_dict_json_reply_never_escapes_as_attribute_error():
    # A model reply of "[]" or "42" is valid JSON but not an object. It must
    # surface as a validation failure (JudgeError / review sentinel), never
    # as an AttributeError that kills a whole batch.
    judge = StubJudge(["[]", "42", "null"])
    with pytest.raises(JudgeError):
        judge.extract_claims("q", "a")

    judge = StubJudge(["[]", "42", "null"])
    jv = judge.verdict(Claim("c0", "x"), SPANS)
    assert jv.confidence == 0.0  # sentinel, routed to review


def test_bool_confidence_and_bool_span_indices_are_rejected():
    good = json.dumps(
        {"verdict": "supported", "supporting_spans": [0], "confidence": 0.9, "rationale": "r"}
    )
    judge = StubJudge(
        [json.dumps({"verdict": "supported", "supporting_spans": [0], "confidence": True}), good]
    )
    assert judge.verdict(Claim("c0", "x"), SPANS).confidence == 0.9

    judge = StubJudge(
        [
            json.dumps({"verdict": "supported", "supporting_spans": [True], "confidence": 0.9}),
            good,
        ]
    )
    assert judge.verdict(Claim("c0", "x"), SPANS).supporting == (0,)


def test_json_object_is_found_amid_prose():
    payload = {"verdict": "supported", "supporting_spans": [], "confidence": 0.8, "rationale": "r"}
    judge = StubJudge([f"Sure! Here is the JSON:\n{json.dumps(payload)}\nHope that helps."])
    jv = judge.verdict(Claim("c0", "x"), SPANS)
    assert jv.verdict == Verdict.SUPPORTED
    assert len(judge.prompts) == 1  # no retry burned on the surrounding prose


def test_http_backoff_retries_on_429_then_succeeds(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from grounded_evals.judge import llm as llm_module

    reply = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    attempts = []
    sleeps = []

    def fake_urlopen(req, timeout):
        attempts.append(req.full_url)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", None, None)
        return io.BytesIO(reply)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)

    judge = LLMJudge(model="m", api_key="k")
    assert judge._chat("hi") == "ok"
    assert len(attempts) == 3
    assert sleeps == [1, 2]  # bounded exponential backoff


def test_http_client_error_fails_fast_without_retry(monkeypatch):
    import urllib.error
    import urllib.request

    attempts = []

    def fake_urlopen(req, timeout):
        attempts.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    judge = LLMJudge(model="m", api_key="k")
    with pytest.raises(JudgeError, match="401"):
        judge._chat("hi")
    assert len(attempts) == 1  # auth errors are not retried
