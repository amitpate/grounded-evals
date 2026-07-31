"""Core data structures.

Every stage of the pipeline consumes and produces these dataclasses, so any
verdict in a report can be traced back to the exact source text it was judged
against. Nothing here depends on a model provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"          # span supports a weaker version of the claim
    UNSUPPORTED = "unsupported"
    NOT_CHECKABLE = "not_checkable"  # opinion, hedge, or no factual content


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    text: str

    def sentences(self) -> list["SourceSpan"]:
        """Naive sentence segmentation; spans carry char offsets into the doc."""
        import re

        spans: list[SourceSpan] = []
        for m in re.finditer(r"[^.!?\n]+[.!?]?", self.text):
            s = m.group().strip()
            if len(s) >= 20:  # skip fragments
                spans.append(SourceSpan(self.doc_id, m.start(), m.end(), s))
        return spans


@dataclass(frozen=True)
class SourceSpan:
    doc_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Citation:
    """A citation as emitted by the generator: claim -> document (or span)."""
    doc_id: str
    quote: Optional[str] = None  # generator-provided quote, if any


@dataclass
class Claim:
    claim_id: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    checkable: bool = True


@dataclass
class Alignment:
    claim_id: str
    spans: list[SourceSpan]
    method: str  # "lexical" | "judge" | "quote"


@dataclass
class ClaimResult:
    claim: Claim
    alignment: Alignment
    verdict: Verdict
    confidence: float
    rationale: str

    @property
    def needs_review(self) -> bool:
        return 0.0 < self.confidence < ClaimResult.REVIEW_THRESHOLD

    REVIEW_THRESHOLD: float = 0.7


@dataclass
class EvalReport:
    results: list[ClaimResult]

    # -- aggregate metrics -------------------------------------------------
    @property
    def checkable(self) -> list[ClaimResult]:
        return [r for r in self.results if r.claim.checkable]

    @property
    def citation_recall(self) -> float:
        """Fraction of checkable claims carrying at least one citation."""
        c = self.checkable
        if not c:
            return 0.0
        return sum(1 for r in c if r.claim.citations) / len(c)

    @property
    def citation_precision(self) -> float:
        """Of cited claims, fraction whose citations support them."""
        cited = [r for r in self.checkable if r.claim.citations]
        if not cited:
            return 0.0
        return sum(1 for r in cited if r.verdict == Verdict.SUPPORTED) / len(cited)

    @property
    def unsupported_rate(self) -> float:
        c = self.checkable
        if not c:
            return 0.0
        return sum(1 for r in c if r.verdict == Verdict.UNSUPPORTED) / len(c)

    @property
    def partial_rate(self) -> float:
        c = self.checkable
        if not c:
            return 0.0
        return sum(1 for r in c if r.verdict == Verdict.PARTIAL) / len(c)

    @property
    def review_queue(self) -> list[ClaimResult]:
        return [r for r in self.results if r.needs_review]

    @property
    def review_load(self) -> float:
        if not self.results:
            return 0.0
        return len(self.review_queue) / len(self.results)

    def summary(self) -> dict:
        return {
            "claims_total": len(self.results),
            "claims_checkable": len(self.checkable),
            "citation_recall": round(self.citation_recall, 3),
            "citation_precision": round(self.citation_precision, 3),
            "unsupported_rate": round(self.unsupported_rate, 3),
            "partial_rate": round(self.partial_rate, 3),
            "review_load": round(self.review_load, 3),
        }

