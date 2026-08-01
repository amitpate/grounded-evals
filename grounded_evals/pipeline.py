"""End-to-end evaluation pipeline.

    extract claims -> resolve citation markers -> align to source spans
    -> verdict -> report

Citation markers: generators cite with markers ("[1]", "[wembley]") that are
not necessarily corpus doc ids. ``evaluate`` resolves markers through a map
built from the corpus — each doc's id plus its 1-based position in the source
list — merged with any caller-provided ``marker_map``. Unresolvable markers
are left as-is and the claim falls back to whole-corpus alignment, so the
failure is visible in the report instead of silently dropped.

Alignment strategy, in order of trust:
1. quote:   the generator provided a quote found in the cited doc (exact, or
            whitespace/case-normalized).
2. lexical: top-k sentences from the cited doc(s) by content-token overlap.
3. corpus:  claim cites nothing / cites an unknown doc -> best spans from the
            whole corpus are offered, so "uncited but true" is distinguishable
            from "uncited and false" in the report.
"""
from __future__ import annotations

import re
from collections import Counter

from ._text import content_tokens
from .judge.base import Judge
from .schema import (
    Alignment,
    Citation,
    Claim,
    ClaimResult,
    EvalReport,
    JudgeVerdict,
    SourceDoc,
    SourceSpan,
    Verdict,
)

TOP_K = 4


def _overlap(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _top_spans(claim: Claim, spans: list[SourceSpan], k: int = TOP_K) -> list[SourceSpan]:
    return sorted(spans, key=lambda s: _overlap(claim.text, s.text), reverse=True)[:k]


def _find_quote(doc: SourceDoc, quote: str) -> SourceSpan | None:
    """Locate a generator-provided quote in the doc, tolerating whitespace
    and case drift; returned offsets always index the original text."""
    start = doc.text.find(quote)
    if start != -1:
        return SourceSpan(doc.doc_id, start, start + len(quote), quote)
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, doc.text, re.IGNORECASE)
    if m:
        return SourceSpan(doc.doc_id, m.start(), m.end(), m.group())
    return None


def marker_map(docs: list[SourceDoc]) -> dict[str, str]:
    """Default marker resolution: doc ids (case-insensitive) and 1-based
    source-list positions ("1" -> first doc), the two conventions grounded
    generators actually emit.

    When the two collide — a corpus whose doc ids are themselves numeric
    strings — the literal doc id wins, deterministically and regardless of
    doc order: "[2]" always means the doc *named* "2" if one exists.
    """
    mapping: dict[str, str] = {}
    for i, d in enumerate(docs):  # positional markers: lowest precedence
        mapping[str(i + 1)] = d.doc_id
    for d in docs:  # literal ids override positions
        mapping[d.doc_id] = d.doc_id
        mapping[d.doc_id.lower()] = d.doc_id
    return mapping


def _resolve(citations: list[Citation], markers: dict[str, str]) -> list[Citation]:
    resolved = []
    for c in citations:
        raw = c.doc_id.strip()
        doc_id = markers.get(raw) or markers.get(raw.lower(), c.doc_id)
        resolved.append(Citation(doc_id=doc_id, quote=c.quote))
    return resolved


def align(claim: Claim, corpus: dict[str, SourceDoc]) -> Alignment:
    cited_docs = [corpus[c.doc_id] for c in claim.citations if c.doc_id in corpus]

    # 1. generator-provided quotes that appear in the cited doc
    for c in claim.citations:
        if c.quote and c.doc_id in corpus:
            span = _find_quote(corpus[c.doc_id], c.quote)
            if span is not None:
                return Alignment(claim.claim_id, [span], method="quote")

    # 2. lexical top-k within cited docs
    if cited_docs:
        spans = [s for d in cited_docs for s in d.sentences()]
        return Alignment(claim.claim_id, _top_spans(claim, spans), method="lexical")

    # 3. fall back to the whole corpus
    spans = [s for d in corpus.values() for s in d.sentences()]
    return Alignment(claim.claim_id, _top_spans(claim, spans), method="corpus")


def evaluate(
    question: str,
    answer: str,
    docs: list[SourceDoc],
    judge: Judge,
    *,
    extra_markers: dict[str, str] | None = None,
) -> EvalReport:
    """Evaluate one grounded answer against its source corpus.

    ``extra_markers`` extends the default marker resolution (see
    ``marker_map``) for generators with bespoke citation labels.

    Raises:
        ValueError: If two docs share a ``doc_id`` — a silent last-wins
            collapse would judge claims against the wrong document.
    """
    corpus = {d.doc_id: d for d in docs}
    if len(corpus) != len(docs):
        counts = Counter(d.doc_id for d in docs)
        dupes = sorted(doc_id for doc_id, n in counts.items() if n > 1)
        raise ValueError(f"duplicate doc_id(s) in corpus: {dupes}")
    markers = marker_map(docs)
    if extra_markers:
        markers.update(extra_markers)

    results: list[ClaimResult] = []
    for claim in judge.extract_claims(question, answer):
        claim.citations = _resolve(claim.citations, markers)
        alignment = align(claim, corpus)
        if not claim.checkable:
            results.append(
                ClaimResult(
                    claim, alignment, Verdict.NOT_CHECKABLE, 1.0, "not a factual claim"
                )
            )
            continue
        jv: JudgeVerdict = judge.verdict(claim, alignment.spans)
        supporting = [
            alignment.spans[i] for i in jv.supporting if 0 <= i < len(alignment.spans)
        ]
        results.append(
            ClaimResult(claim, alignment, jv.verdict, jv.confidence, jv.rationale, supporting)
        )

    return EvalReport(results)
