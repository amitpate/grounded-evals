"""Deterministic judge for CI and tests.

Claim extraction: sentence-split the answer (decimal-safe), attach citation
markers found in each sentence. Verdicts: token-overlap entailment proxy with
greedy multi-span union, so claims synthesized from several source sentences
can still be SUPPORTED. Numerals are checked first and normalized
(90,000 == 90000): a claim whose number is missing from its supporting spans
is at most PARTIAL, and a claim whose supporting spans state a *different*
number is CONTRADICTED — numeric substitution is the sneakiest failure and the
one thing a lexical proxy can reliably catch.

Deterministic, fast, and intentionally dumb: it exists so the pipeline's
plumbing and metrics are testable without a model, and so judge-agreement
numbers have a floor to beat.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from .._text import content_tokens, numerals
from ..schema import Citation, Claim, JudgeVerdict, SourceSpan, Verdict

SUPPORT_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.45

# Sentence + trailing citation markers; a '.' between digits is not a boundary.
_CLAIM_RE = re.compile(r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+[.!?]*(?:\s*\[[^\]]+\])*")
_MARKER_RE = re.compile(r"\[([^\]]+)\]")
_LEADING_MARKERS_RE = re.compile(r"^(?:\s*\[[^\]]+\])+")
_HEDGE_RE = re.compile(r"(?i)^(i think|perhaps|arguably|in my view|generally)")


class MockJudge:
    """Deterministic Judge implementation; see module docstring."""

    fingerprint = {"judge": "mock", "version": "2"}

    def extract_claims(self, question: str, answer: str) -> list[Claim]:
        claims: list[Claim] = []
        for m in _CLAIM_RE.finditer(answer):
            sent = m.group().strip()
            # Markers trailing the terminator ("... Kingdom. [1]") belong to
            # THIS sentence; markers leading a sentence belong to the PREVIOUS
            # claim — the sweep below handles both.
            leading = _LEADING_MARKERS_RE.match(sent)
            if leading and claims:
                for marker in _MARKER_RE.findall(leading.group()):
                    claims[-1].citations.append(Citation(doc_id=marker))
                sent = sent[leading.end():].strip()
            cites = [Citation(doc_id=c) for c in _MARKER_RE.findall(sent)]
            text = _MARKER_RE.sub("", sent).strip()
            text = re.sub(r"\s{2,}", " ", text)
            if len(text) < 15:
                # Too short to be a claim ("Yes.") — but its trailing markers
                # still bind to the nearest surviving claim rather than
                # silently vanishing with the fragment.
                if cites and claims:
                    claims[-1].citations.extend(cites)
                continue
            claims.append(
                Claim(
                    claim_id=f"c{len(claims)}",
                    text=text,
                    citations=cites,
                    checkable=not _HEDGE_RE.match(text),
                )
            )
        return claims

    def verdict(self, claim: Claim, spans: Sequence[SourceSpan]) -> JudgeVerdict:
        ctoks = content_tokens(claim.text)
        if not ctoks:
            return JudgeVerdict(Verdict.NOT_CHECKABLE, 1.0, "no factual content tokens")

        # Greedy union: add spans by marginal token coverage so a claim
        # synthesized from several source sentences can reach SUPPORTED.
        overlaps = [(len(ctoks & content_tokens(sp.text)), i) for i, sp in enumerate(spans)]
        overlaps.sort(key=lambda t: (-t[0], t[1]))
        covered: set[str] = set()
        used: list[int] = []
        for _, i in overlaps:
            gain = (ctoks & content_tokens(spans[i].text)) - covered
            if not gain:
                continue
            covered |= gain
            used.append(i)
            if len(covered) / len(ctoks) >= SUPPORT_THRESHOLD:
                break
        coverage = len(covered) / len(ctoks)

        if coverage < PARTIAL_THRESHOLD:
            return JudgeVerdict(
                Verdict.UNSUPPORTED,
                min(0.9, 1.0 - coverage),
                f"no spans cover the claim (best union {coverage:.0%})",
            )

        claim_nums = numerals(claim.text)
        span_nums = numerals(" ".join(spans[i].text for i in used))
        missing = claim_nums - span_nums
        if missing:
            # The greedy loop breaks at the coverage threshold, so a numeral
            # may live in a relevant span it never added. Rescue such spans
            # before judging a number missing or conflicting — otherwise a
            # claim can be called CONTRADICTED while a span on the judge's
            # own desk states the exact number.
            for j, sp in enumerate(spans):
                if j in used:
                    continue
                sp_nums = numerals(sp.text)
                if missing & sp_nums and (ctoks & content_tokens(sp.text)):
                    used.append(j)
                    span_nums |= sp_nums
                    missing = claim_nums - span_nums
                    if not missing:
                        break
        supporting = tuple(used)
        if missing:
            # Numbers are load-bearing. A competing number in the supporting
            # spans means the source states a different value: contradicted.
            competing = span_nums - claim_nums
            shown = ", ".join(sorted(missing))
            if competing:
                return JudgeVerdict(
                    Verdict.CONTRADICTED,
                    0.65,
                    f"numeric conflict: claim states {shown}, supporting spans "
                    f"state {', '.join(sorted(competing))}",
                    supporting,
                )
            return JudgeVerdict(
                Verdict.PARTIAL,
                0.6,
                f"numeric gap: claim states {shown}, unattested in supporting spans",
                supporting,
            )

        if coverage >= SUPPORT_THRESHOLD:
            return JudgeVerdict(
                Verdict.SUPPORTED,
                min(0.95, coverage),
                f"spans jointly cover {coverage:.0%} of claim tokens",
                supporting,
            )
        return JudgeVerdict(
            Verdict.PARTIAL,
            0.6,
            f"spans cover only {coverage:.0%} of claim tokens",
            supporting,
        )
