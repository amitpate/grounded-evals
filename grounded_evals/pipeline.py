"""End-to-end evaluation pipeline.

    extract claims -> align citations to source spans -> verdict -> report

Alignment strategy, in order of trust:
1. quote:   the generator provided a quote and it appears in the cited doc.
2. lexical: top-k sentences from the cited doc by content-token overlap.
3. corpus:  claim cites nothing / cites an unknown doc -> best spans from the
            whole corpus are offered, so "uncited but true" is distinguishable
            from "uncited and false" in the report.
"""
from __future__ import annotations

from .judge.base import Judge
from .schema import (
    Alignment,
    Claim,
    ClaimResult,
    EvalReport,
    SourceDoc,
    SourceSpan,
    Verdict,
)

TOP_K = 4


def _overlap(a: str, b: str) -> float:
    from .judge.mock import _content_tokens

    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _top_spans(claim: Claim, spans: list[SourceSpan], k: int = TOP_K) -> list[SourceSpan]:
    return sorted(spans, key=lambda s: _overlap(claim.text, s.text), reverse=True)[:k]


def align(claim: Claim, corpus: dict[str, SourceDoc]) -> Alignment:
    cited_docs = [corpus[c.doc_id] for c in claim.citations if c.doc_id in corpus]

    # 1. generator-provided quotes that actually appear in the cited doc
    for c in claim.citations:
        if c.quote and c.doc_id in corpus and c.quote in corpus[c.doc_id].text:
            doc = corpus[c.doc_id]
            start = doc.text.index(c.quote)
            return Alignment(
                claim.claim_id,
                [SourceSpan(doc.doc_id, start, start + len(c.quote), c.quote)],
                method="quote",
            )

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
) -> EvalReport:
    corpus = {d.doc_id: d for d in docs}
    results: list[ClaimResult] = []

    for claim in judge.extract_claims(question, answer):
        alignment = align(claim, corpus)
        if not claim.checkable:
            results.append(
                ClaimResult(claim, alignment, Verdict.NOT_CHECKABLE, 1.0, "not a factual claim")
            )
            continue
        verdict, confidence, rationale = judge.verdict(claim, alignment.spans)
        results.append(ClaimResult(claim, alignment, verdict, confidence, rationale))

    return EvalReport(results)

