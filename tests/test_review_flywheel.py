"""Contract tests for the review flywheel: adjudication independence and the
not_checkable path from queue to fixtures."""
from grounded_evals.judge import MockJudge
from grounded_evals.review import ingest_adjudications, to_fixture_rows
from grounded_evals.runner import run_dataset

_DOC = {
    "doc_id": "d1",
    "title": "Fact sheet",
    "text": (
        "The Wembley arch spans 315 metres across the stadium. "
        "The stadium seats 90,000 people and reopened in 2007."
    ),
}

_ROWS = [
    {
        "id": "r1",
        "question": "q",
        "answer": "The stadium reopened in 2010. [d1] The arch spans 315 metres. [d1]",
        "docs": [_DOC],
    }
]


def _adjudication(claim_id: str, gold: str) -> dict:
    return {"row_id": "r1", "claim_id": claim_id, "gold": gold}


def test_ingest_returns_fully_independent_copy():
    batch = run_dataset(_ROWS, MockJudge())
    adjudicated = ingest_adjudications(batch, [_adjudication("c0", "contradicted")])

    # Mutating any result in the new batch must not touch the original —
    # including results that were NOT adjudicated.
    original_conf = batch.rows[0].report.results[1].confidence
    adjudicated.rows[0].report.results[1].confidence = 0.123
    assert batch.rows[0].report.results[1].confidence == original_conf
    assert adjudicated.rows[0].report.results[0].rationale == "human adjudication"
    assert batch.rows[0].report.results[0].rationale != "human adjudication"


def test_not_checkable_adjudication_leaves_metric_denominators():
    batch = run_dataset(_ROWS, MockJudge())
    checkable_before = len(batch.rows[0].report.checkable)

    adjudicated = ingest_adjudications(batch, [_adjudication("c0", "not_checkable")])
    assert len(adjudicated.rows[0].report.checkable) == checkable_before - 1
    # The original is untouched.
    assert len(batch.rows[0].report.checkable) == checkable_before


def test_not_checkable_adjudications_are_skipped_in_fixture_export():
    batch = run_dataset(_ROWS, MockJudge())
    rows = to_fixture_rows(
        batch,
        [_adjudication("c0", "not_checkable"), _adjudication("c1", "supported")],
    )
    # Only the checkable-verdict adjudication becomes a fixture, so the
    # exported rows always load back through agreement.load_fixtures.
    assert [r["gold"] for r in rows] == ["supported"]
    assert rows[0]["tags"] == ["adjudicated"]
