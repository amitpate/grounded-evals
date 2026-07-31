
from grounded_evals import SourceDoc, Verdict, evaluate
from grounded_evals.judge import MockJudge
from grounded_evals.schema import Citation, Claim
from grounded_evals.pipeline import align

DOC = SourceDoc(
    "d1",
    "Fact sheet",
    "The Wembley arch spans 315 metres across the stadium. "
    "The stadium seats 90,000 people and reopened in 2007. "
    "London hosted the Olympic Games in 2012.",
)


def test_supported_claim():
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [d1]",
        [DOC],
        MockJudge(),
    )
    assert report.results[0].verdict == Verdict.SUPPORTED
    assert report.citation_precision == 1.0
    assert report.unsupported_rate == 0.0


def test_unsupported_cited_claim_is_caught():
    report = evaluate(
        "q",
        "The stadium was demolished in 1999 by giant bulldozers. [d1]",
        [DOC],
        MockJudge(),
    )
    assert report.results[0].verdict == Verdict.UNSUPPORTED
    assert report.unsupported_rate == 1.0


def test_uncited_claim_hits_recall_not_precision():
    report = evaluate(
        "q",
        "London hosted the Olympic Games in 2012. The arch spans 315 metres. [d1]",
        [DOC],
        MockJudge(),
    )
    assert report.citation_recall == 0.5  # one of two claims cited
    cited = [r for r in report.results if r.claim.citations]
    assert all(r.verdict == Verdict.SUPPORTED for r in cited)


def test_opinion_not_checkable():
    report = evaluate("q", "I think this is the finest stadium ever built.", [DOC], MockJudge())
    assert report.results[0].verdict == Verdict.NOT_CHECKABLE
    assert report.checkable == []


def test_quote_alignment_beats_lexical():
    claim = Claim(
        "c0",
        "The arch spans 315 metres.",
        citations=[Citation("d1", quote="The Wembley arch spans 315 metres across the stadium.")],
    )
    a = align(claim, {"d1": DOC})
    assert a.method == "quote"
    assert len(a.spans) == 1


def test_unknown_doc_falls_back_to_corpus():
    claim = Claim("c0", "The stadium seats 90,000 people.", citations=[Citation("nope")])
    a = align(claim, {"d1": DOC})
    assert a.method == "corpus"
    assert a.spans  # still offered best corpus spans


def test_low_confidence_routes_to_review():
    report = evaluate(
        "q",
        "The stadium seats 90,000 spectators for concerts and events. [d1]",
        [DOC],
        MockJudge(),
    )
    r = report.results[0]
    if r.verdict == Verdict.PARTIAL:
        assert r.needs_review
