from grounded_evals import SourceDoc, Verdict, evaluate
from grounded_evals.judge import MockJudge
from grounded_evals.pipeline import align
from grounded_evals.schema import Alignment, Citation, Claim, ClaimResult, EvalReport

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
    assert report.ungrounded_rate == 0.0


def test_unsupported_cited_claim_is_caught():
    report = evaluate(
        "q",
        "The stadium was demolished in 1999 by giant bulldozers. [d1]",
        [DOC],
        MockJudge(),
    )
    assert report.results[0].verdict == Verdict.UNSUPPORTED
    assert report.unsupported_rate == 1.0
    assert report.ungrounded_rate == 1.0


def test_numeric_conflict_is_contradicted_not_partial():
    # The source says 2007; the claim says 2010. A competing number in the
    # supporting span is a contradiction, not a gap.
    report = evaluate(
        "q",
        "The stadium reopened in 2010. [d1]",
        [DOC],
        MockJudge(),
    )
    r = report.results[0]
    assert r.verdict == Verdict.CONTRADICTED
    assert r.needs_review  # the sneakiest class gets human eyes
    assert report.contradicted_rate == 1.0


def test_multi_span_synthesis_is_supported():
    # Claim combines two source sentences; joint (union) coverage must win.
    report = evaluate(
        "q",
        "The arch spans 315 metres and the stadium seats 90,000 people. [d1]",
        [DOC],
        MockJudge(),
    )
    r = report.results[0]
    assert r.verdict == Verdict.SUPPORTED
    assert len({s.text for s in r.supporting_spans}) == 2


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
    assert report.abstained


def test_numbered_marker_resolves_to_source_list_position():
    # Generators cite "[1]" meaning the first doc in the source list.
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [1]",
        [DOC],
        MockJudge(),
    )
    r = report.results[0]
    assert r.claim.citations[0].doc_id == "d1"
    assert r.alignment.method == "lexical"
    assert r.verdict == Verdict.SUPPORTED


def test_quote_alignment_beats_lexical():
    claim = Claim(
        "c0",
        "The arch spans 315 metres.",
        citations=[Citation("d1", quote="The Wembley arch spans 315 metres across the stadium.")],
    )
    a = align(claim, {"d1": DOC})
    assert a.method == "quote"
    assert len(a.spans) == 1


def test_fuzzy_quote_survives_whitespace_and_case_drift():
    claim = Claim(
        "c0",
        "The arch spans 315 metres.",
        citations=[Citation("d1", quote="the wembley arch  spans 315 metres across the stadium")],
    )
    a = align(claim, {"d1": DOC})
    assert a.method == "quote"
    span = a.spans[0]
    assert DOC.text[span.start:span.end] == span.text  # offsets index the original


def test_unknown_doc_falls_back_to_corpus():
    claim = Claim("c0", "The stadium seats 90,000 people.", citations=[Citation("nope")])
    a = align(claim, {"d1": DOC})
    assert a.method == "corpus"
    assert a.spans  # still offered best corpus spans


def test_low_confidence_partial_routes_to_review():
    report = evaluate(
        "q",
        "The stadium seats 90,000 spectators for concerts and events. [d1]",
        [DOC],
        MockJudge(),
    )
    r = report.results[0]
    assert r.verdict == Verdict.PARTIAL
    assert r.needs_review


def test_judge_failure_sentinel_confidence_zero_routes_to_review():
    r = ClaimResult(
        Claim("c0", "any factual claim"),
        Alignment("c0", [], "lexical"),
        Verdict.UNSUPPORTED,
        0.0,
        "judge failure -> review",
    )
    assert r.needs_review


def test_sentence_offsets_round_trip_exactly():
    for span in DOC.sentences():
        assert DOC.text[span.start:span.end] == span.text


def test_decimal_numbers_do_not_split_sentences_or_drop_text():
    doc = SourceDoc("d", "t", "The arch spans 315.5 metres above the bowl. It is tall.")
    spans = doc.sentences()
    assert [s.text for s in spans] == [
        "The arch spans 315.5 metres above the bowl.",
        "It is tall.",
    ]
    for span in spans:
        assert doc.text[span.start:span.end] == span.text


def test_empty_report_metrics_are_none_not_zero():
    report = EvalReport(results=[])
    assert report.citation_precision is None
    assert report.citation_recall is None
    assert report.ungrounded_rate is None
    assert report.summary()["citation_precision"] is None


def test_per_citation_precision_sees_overcitation():
    other = SourceDoc("d2", "Other", "Something entirely unrelated to stadiums here.")
    # Claim cites both docs; only d1 contributes a supporting span.
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [d1] [d2]",
        [DOC, other],
        MockJudge(),
    )
    assert report.citation_precision == 1.0  # claim-level: supported
    assert report.per_citation_precision == 0.5  # citation-level: d2 was padding


def test_per_citation_precision_not_inflatable_by_repeating_a_citation():
    other = SourceDoc("d2", "Other", "Something entirely unrelated to stadiums here.")
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [d1] [d1] [d1] [d2]",
        [DOC, other],
        MockJudge(),
    )
    # Duplicates of the good citation count once: still 1 good of 2 distinct.
    assert report.per_citation_precision == 0.5


def test_marker_collision_literal_doc_id_beats_position():
    # Corpus doc ids are themselves numeric and out of order: "[2]" must mean
    # the doc NAMED "2", not the second doc in the list.
    docs = [
        SourceDoc("2", "Named two", "The stadium seats 90,000 people and reopened in 2007."),
        SourceDoc("1", "Named one", "Something entirely unrelated to stadiums here."),
    ]
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [2]",
        docs,
        MockJudge(),
    )
    r = report.results[0]
    assert r.claim.citations[0].doc_id == "2"
    assert r.verdict == Verdict.SUPPORTED


def test_extra_markers_resolve_custom_labels():
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. [srcA]",
        [DOC],
        MockJudge(),
        extra_markers={"srcA": "d1"},
    )
    r = report.results[0]
    assert r.claim.citations[0].doc_id == "d1"
    assert r.alignment.method == "lexical"


def test_duplicate_doc_ids_raise_instead_of_silently_collapsing():
    import pytest

    with pytest.raises(ValueError, match="duplicate doc_id"):
        evaluate("q", "any answer", [DOC, SourceDoc("d1", "Copy", "Other text.")], MockJudge())


def test_mock_rescues_numeral_from_span_greedy_union_skipped():
    # The greedy union hits its coverage threshold on span 0 (which carries a
    # competing numeral), but span 1 states the claim's exact number. The
    # verdict must be SUPPORTED, not a false CONTRADICTED.
    from grounded_evals.schema import SourceSpan

    def span(start, text):
        return SourceSpan("d1", start, start + len(text), text)

    spans = [
        span(0, "The stadium, built in 1923, seats 90,000 people and reopened for stadium events."),
        span(100, "The stadium reopened in 2007."),
    ]
    claim = Claim("c0", "The stadium seats 90,000 people and reopened in 2007.")
    jv = MockJudge().verdict(claim, spans)
    assert jv.verdict == Verdict.SUPPORTED
    assert set(jv.supporting) == {0, 1}


def test_short_fragment_citations_bind_to_previous_claim():
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007. Yes. [d1]",
        [DOC],
        MockJudge(),
    )
    assert len(report.results) == 1  # "Yes." is not a claim
    assert [c.doc_id for c in report.results[0].claim.citations] == ["d1"]


def test_leading_markers_bind_to_previous_claim():
    report = evaluate(
        "q",
        "The stadium seats 90,000 people and reopened in 2007.\n[d1] The arch spans 315 metres.",
        [DOC],
        MockJudge(),
    )
    first, second = report.results
    assert [c.doc_id for c in first.claim.citations] == ["d1"]
    assert second.claim.citations == []
    assert second.claim.text.startswith("The arch")
