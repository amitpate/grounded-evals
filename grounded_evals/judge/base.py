"""Judge protocol.

A judge answers two structured questions:
1. extract_claims: decompose an answer into atomic, checkable claims.
2. verdict: given a claim and candidate source spans, return
   (verdict, confidence, rationale).

Implementations must return structured output conforming to the schemas in
PROMPTS; callers validate and retry on mismatch. The mock judge is
deterministic so the full pipeline runs in CI with no keys.
"""
from __future__ import annotations

from typing import Protocol

from ..schema import Claim, SourceSpan, Verdict


class Judge(Protocol):
    def extract_claims(self, question: str, answer: str) -> list[Claim]:
        ...

    def verdict(
        self, claim: Claim, spans: list[SourceSpan]
    ) -> tuple[Verdict, float, str]:
        ...


EXTRACT_PROMPT = """\
Decompose the ANSWER into atomic factual claims.

Rules:
- One checkable assertion per claim; split conjunctions.
- Keep the claim's citation markers (e.g. [1], [2]) attached to the claim.
- Mark opinions, hedges and meta-commentary as checkable=false.
- Do not paraphrase away specificity (numbers, names, dates survive verbatim).

Return JSON: {"claims": [{"text": str, "citation_ids": [int], "checkable": bool}]}

QUESTION: {question}
ANSWER: {answer}
"""

VERDICT_PROMPT = """\
You are verifying whether SOURCE SPANS support a CLAIM. Judge only what the
spans state — not what is plausible, commonly known, or implied beyond them.

Verdicts:
- supported: a span (or spans jointly) entail the claim as written.
- partial: spans entail a strictly weaker version (missing a number, a scope
  qualifier, or half of a conjunction). State exactly what is missing.
- unsupported: no span entails the claim, or spans contradict it.

Return JSON: {"verdict": "supported|partial|unsupported",
              "confidence": float 0..1,
              "rationale": str}

CLAIM: {claim}
SOURCE SPANS:
{spans}
"""

