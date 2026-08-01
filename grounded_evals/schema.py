"""Core data structures.

Every stage of the pipeline consumes and produces these dataclasses, so any
verdict in a report can be traced back to the exact source text it was judged
against. Nothing here depends on a model provider.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class Verdict(str, Enum):
    """Per-claim faithfulness verdict.

    The taxonomy follows what the attribution literature converged on
    (AttrScore, CAQA, RAGTruth): *contradicted* (the source states otherwise)
    and *unsupported* (the source is silent) are different product failures —
    the first breaks trust in the source link, the second is a retrieval or
    citation gap — and are never collapsed into one class.
    """

    SUPPORTED = "supported"
    PARTIAL = "partial"              # spans support a strictly weaker claim
    CONTRADICTED = "contradicted"    # spans state something incompatible
    UNSUPPORTED = "unsupported"      # spans are silent on the claim
    NOT_CHECKABLE = "not_checkable"  # opinion, hedge, or no factual content


# A '.' between two digits is a decimal point, not a sentence terminator.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|(?<=\d)\.(?=\d))+[.!?]*")


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    text: str

    def sentences(self) -> list[SourceSpan]:
        """Sentence spans whose char offsets round-trip exactly.

        Invariant: ``doc.text[span.start:span.end] == span.text`` for every
        returned span. Decimal numbers ("315.5") do not split a sentence, and
        no source text is dropped beyond whitespace and bare punctuation
        runs — a verifier must never silently lose evidence.
        """
        spans: list[SourceSpan] = []
        for m in _SENTENCE_RE.finditer(self.text):
            raw = m.group()
            stripped = raw.strip()
            if not stripped:
                continue
            start = m.start() + (len(raw) - len(raw.lstrip()))
            spans.append(SourceSpan(self.doc_id, start, start + len(stripped), stripped))
        return spans


@dataclass(frozen=True)
class SourceSpan:
    """A contiguous slice of a source document; text matches the offsets."""

    doc_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Citation:
    """A citation as emitted by the generator.

    ``doc_id`` may initially be a raw marker ("1", "wembley"); the pipeline
    resolves markers to corpus doc ids before alignment (see
    ``pipeline.evaluate``). ``quote`` is a generator-provided quote, if any.
    """

    doc_id: str
    quote: str | None = None


@dataclass
class Claim:
    claim_id: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    checkable: bool = True


@dataclass
class Alignment:
    """Claim -> candidate source spans, and how they were chosen."""

    claim_id: str
    spans: list[SourceSpan]
    method: str  # "quote" | "lexical" | "corpus"


@dataclass(frozen=True)
class JudgeVerdict:
    """What a judge returns for one (claim, spans) query.

    ``supporting`` holds indices into the span list the judge was shown —
    the minimal set of spans the verdict rests on ([] when nothing supports).
    A confidence of exactly 0.0 is the judge-failure sentinel and always
    routes to review.
    """

    verdict: Verdict
    confidence: float
    rationale: str
    supporting: tuple[int, ...] = ()


@dataclass
class ClaimResult:
    REVIEW_THRESHOLD: ClassVar[float] = 0.7

    claim: Claim
    alignment: Alignment
    verdict: Verdict
    confidence: float
    rationale: str
    supporting_spans: list[SourceSpan] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """Whether this verdict falls below the automation boundary.

        Confidence 0.0 (the judge-failure sentinel) is deliberately included:
        a failed verdict must never bypass review.
        """
        return self.claim.checkable and self.confidence < ClaimResult.REVIEW_THRESHOLD


def _ratio(num: int, denom: int) -> float | None:
    """None when there is nothing to measure — never a fake 0.0."""
    return num / denom if denom else None


@dataclass
class EvalReport:
    results: list[ClaimResult]

    # -- aggregate metrics -------------------------------------------------
    @property
    def checkable(self) -> list[ClaimResult]:
        return [r for r in self.results if r.claim.checkable]

    @property
    def abstained(self) -> bool:
        """True when the answer contained no checkable factual claims.

        On questions the corpus cannot answer, abstaining is the *correct*
        behavior; batch runners use this to score abstention quality.
        """
        return not self.checkable

    @property
    def citation_recall(self) -> float | None:
        """Fraction of checkable claims carrying at least one citation."""
        c = self.checkable
        return _ratio(sum(1 for r in c if r.claim.citations), len(c))

    @property
    def citation_precision(self) -> float | None:
        """Of cited checkable claims, fraction fully supported by their sources."""
        cited = [r for r in self.checkable if r.claim.citations]
        return _ratio(
            sum(1 for r in cited if r.verdict is Verdict.SUPPORTED), len(cited)
        )

    @property
    def per_citation_precision(self) -> float | None:
        """Of individual citations, fraction that contributed a supporting span.

        Distinct from ``citation_precision``: a claim citing [1][2] where only
        [1] supports it is over-citing, and this metric sees it (ALCE-style
        per-citation precision). Duplicate citations of the same doc on one
        claim count once — repeating a good citation must not buy score.
        """
        pairs = [
            (r, doc_id)
            for r in self.checkable
            for doc_id in sorted({c.doc_id for c in r.claim.citations})
        ]
        good = sum(
            1
            for r, doc_id in pairs
            if r.verdict in (Verdict.SUPPORTED, Verdict.PARTIAL)
            and doc_id in {s.doc_id for s in r.supporting_spans}
        )
        return _ratio(good, len(pairs))

    def _rate(self, verdict: Verdict) -> float | None:
        c = self.checkable
        return _ratio(sum(1 for r in c if r.verdict is verdict), len(c))

    @property
    def partial_rate(self) -> float | None:
        return self._rate(Verdict.PARTIAL)

    @property
    def contradicted_rate(self) -> float | None:
        return self._rate(Verdict.CONTRADICTED)

    @property
    def unsupported_rate(self) -> float | None:
        return self._rate(Verdict.UNSUPPORTED)

    @property
    def ungrounded_rate(self) -> float | None:
        """Contradicted + unsupported: the headline trust-erosion number."""
        c = self.checkable
        return _ratio(
            sum(
                1
                for r in c
                if r.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED)
            ),
            len(c),
        )

    @property
    def review_queue(self) -> list[ClaimResult]:
        return [r for r in self.results if r.needs_review]

    @property
    def review_load(self) -> float | None:
        return _ratio(len(self.review_queue), len(self.results))

    def summary(self) -> dict:
        def rnd(x: float | None) -> float | None:
            return None if x is None else round(x, 3)

        return {
            "claims_total": len(self.results),
            "claims_checkable": len(self.checkable),
            "abstained": self.abstained,
            "citation_recall": rnd(self.citation_recall),
            "citation_precision": rnd(self.citation_precision),
            "per_citation_precision": rnd(self.per_citation_precision),
            "partial_rate": rnd(self.partial_rate),
            "contradicted_rate": rnd(self.contradicted_rate),
            "unsupported_rate": rnd(self.unsupported_rate),
            "ungrounded_rate": rnd(self.ungrounded_rate),
            "review_load": rnd(self.review_load),
        }
