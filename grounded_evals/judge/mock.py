"""Deterministic judge for CI and tests.

Claim extraction: sentence-split the answer, attach citation markers found in
each sentence. Verdicts: token-overlap entailment proxy — supported when the
best span covers most of the claim's content tokens, partial on moderate
overlap, unsupported otherwise. Deterministic, fast, and intentionally dumb:
it exists so the pipeline's plumbing and metrics are testable without a model,
and so judge-agreement numbers have a floor to beat.
"""
from __future__ import annotations

import re

from ..schema import Citation, Claim, SourceSpan, Verdict

_STOP = frozenset(
    "a an the of to in on for and or is are was were be been with by at as it its "
    "this that from into over under after before not no".split()
)


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9']+", text)
        if t.lower() not in _STOP and len(t) > 2
    }


class MockJudge:
    def extract_claims(self, question: str, answer: str) -> list[Claim]:
        claims: list[Claim] = []
        for i, m in enumerate(re.finditer(r"[^.!?\n]+[.!?]?(?:\s*\[[^\]]+\])*", answer)):
            sent = m.group().strip()
            # citations trailing the sentence terminator ("... Kingdom. [doc]")
            # belong to THIS sentence, and leading markers belong to the
            # previous claim — the sweep below handles both.
            leading = re.match(r"^(?:\s*\[[^\]]+\])+", sent)
            if leading and claims:
                for c in re.findall(r"\[([^\]]+)\]", leading.group()):
                    claims[-1].citations.append(Citation(doc_id=c))
                sent = sent[leading.end():].strip()
            if len(sent) < 15:
                continue
            cites = [Citation(doc_id=c) for c in re.findall(r"\[([^\]]+)\]", sent)]
            text = re.sub(r"\s*\[[^\]]+\]", "", sent).strip()
            checkable = not re.match(
                r"(?i)^(i think|perhaps|arguably|in my view|generally)", text
            )
            claims.append(
                Claim(claim_id=f"c{i}", text=text, citations=cites, checkable=bool(checkable))
            )
        return claims

    def verdict(
        self, claim: Claim, spans: list[SourceSpan]
    ) -> tuple[Verdict, float, str]:
        ctoks = _content_tokens(claim.text)
        if not ctoks:
            return Verdict.NOT_CHECKABLE, 1.0, "no factual content tokens"
        best, best_span = 0.0, None
        for sp in spans:
            cover = len(ctoks & _content_tokens(sp.text)) / len(ctoks)
            if cover > best:
                best, best_span = cover, sp
        # numbers are load-bearing: a claim whose numerals don't all appear in
        # the best span can be at most PARTIAL, whatever the token overlap.
        num_re = r"\d+(?:,\d{3})*(?:\.\d+)?"
        claim_nums = set(re.findall(num_re, claim.text))
        span_nums = set(re.findall(num_re, best_span.text)) if best_span else set()
        if best >= 0.45 and claim_nums - span_nums:
            missing = ", ".join(sorted(claim_nums - span_nums))
            return Verdict.PARTIAL, 0.6, f"numeric mismatch: claim states {missing}, best span does not ({best_span.text[:70]!r})"
        if best >= 0.75:
            return Verdict.SUPPORTED, min(0.95, best), f"span covers {best:.0%} of claim tokens: {best_span.text[:80]!r}"
        if best >= 0.45:
            return Verdict.PARTIAL, 0.6, f"best span covers only {best:.0%} of claim tokens"
        return Verdict.UNSUPPORTED, min(0.9, 1.0 - best), f"no span covers claim (best {best:.0%})"

