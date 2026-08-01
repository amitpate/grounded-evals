import json

import pytest

from grounded_evals import Verdict, __version__
from grounded_evals.judge import MockJudge
from grounded_evals.judge.base import JudgeError
from grounded_evals.review import (
    export_queue,
    ingest_adjudications,
    to_fixture_rows,
    write_queue,
)
from grounded_evals.runner import run_dataset

WEMBLEY_DOC = {
    "doc_id": "wembley",
    "title": "Wembley Stadium factsheet",
    "text": (
        "Wembley Stadium in London has a seating capacity of 90,000, making it "
        "the largest stadium in the United Kingdom. The stadium reopened in "
        "2007 after a complete rebuild."
    ),
}

GROUNDED_ANSWER = "The stadium reopened in 2007 after a complete rebuild. [1]"
CONTRADICTED_ANSWER = "The stadium reopened in 2010 after a complete rebuild. [1]"
ABSTAINING_ANSWER = "I think the sources do not say who designed the stadium."


def make_row(row_id, answer, *, answerable=True):
    row = {
        "id": row_id,
        "question": "What is notable about Wembley Stadium?",
        "answer": answer,
        "docs": [WEMBLEY_DOC],
    }
    if not answerable:
        row["answerable"] = False
    return row


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class ExplodingJudge(MockJudge):
    """MockJudge whose extraction fails when the answer contains BOOM."""

    def extract_claims(self, question, answer):
        if "BOOM" in answer:
            raise JudgeError("synthetic extraction failure")
        return super().extract_claims(question, answer)


# -- dataset loading and aggregation ----------------------------------------
def test_dataset_jsonl_round_trip(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(
        path, [make_row("good", GROUNDED_ANSWER), make_row("bad", CONTRADICTED_ANSWER)]
    )
    batch = run_dataset(path, MockJudge())

    assert [r.row_id for r in batch.rows] == ["good", "bad"]
    assert batch.errors == []
    assert batch.rows[0].report.results[0].verdict is Verdict.SUPPORTED
    assert batch.rows[1].report.results[0].verdict is Verdict.CONTRADICTED
    assert batch.total_claims == 2
    assert batch.review_queue_size == 1  # only the contradiction is low-confidence
    assert batch.review_load == 0.5
    assert batch.mean_metrics["mean_contradicted_rate"] == 0.5
    assert batch.mean_metrics["mean_citation_precision"] == 0.5
    assert batch.judge_fingerprint == {"judge": "mock", "version": "2"}


def test_invalid_json_names_line_number(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(make_row("ok", GROUNDED_ANSWER)) + "\nnot json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="line 2"):
        run_dataset(path, MockJudge())


def test_missing_field_names_line_number(tmp_path):
    bad = make_row("bad", GROUNDED_ANSWER)
    del bad["answer"]
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [make_row("ok", GROUNDED_ANSWER), bad])
    with pytest.raises(ValueError, match="line 2"):
        run_dataset(path, MockJudge())


def test_abstention_quality_over_mixed_rows():
    rows = [
        make_row("refuses-correctly", ABSTAINING_ANSWER, answerable=False),
        make_row("hallucinates", CONTRADICTED_ANSWER, answerable=False),
        make_row("answers", GROUNDED_ANSWER),
    ]
    batch = run_dataset(rows, MockJudge())
    assert batch.rows[0].report.abstained
    assert not batch.rows[1].report.abstained
    assert batch.correct_refusal_rate == 0.5  # one of two unanswerable rows refused
    assert batch.false_refusal_rate == 0.0  # the answerable row did not refuse


def test_abstention_rates_none_without_matching_rows():
    batch = run_dataset([make_row("answers", GROUNDED_ANSWER)], MockJudge())
    assert batch.correct_refusal_rate is None  # no unanswerable rows to score
    assert batch.false_refusal_rate == 0.0


def test_errored_row_is_recorded_not_fatal():
    rows = [
        make_row("ok", GROUNDED_ANSWER),
        make_row("broken", "BOOM. " + GROUNDED_ANSWER),
    ]
    batch = run_dataset(rows, ExplodingJudge())
    assert [r.row_id for r in batch.rows] == ["ok"]
    assert [e.row_id for e in batch.errors] == ["broken"]
    assert "synthetic extraction failure" in batch.errors[0].error
    assert batch.to_dict()["dataset_rows"] == 2


def test_parallel_matches_serial():
    rows = [
        make_row("good", GROUNDED_ANSWER),
        make_row("bad", CONTRADICTED_ANSWER),
        make_row("refuses", ABSTAINING_ANSWER, answerable=False),
        make_row("also-good", GROUNDED_ANSWER),
        make_row("broken", "BOOM. " + GROUNDED_ANSWER),
    ]
    serial = run_dataset(rows, ExplodingJudge(), max_workers=1)
    parallel = run_dataset(rows, ExplodingJudge(), max_workers=2)
    assert serial.to_dict() == parallel.to_dict()


def test_save_writes_loadable_self_contained_json(tmp_path):
    batch = run_dataset([make_row("good", GROUNDED_ANSWER)], MockJudge())
    out = tmp_path / "report.json"
    batch.save(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["grounded_evals_version"] == __version__
    assert data["judge_fingerprint"] == {"judge": "mock", "version": "2"}
    assert data["dataset_rows"] == 1
    assert data["aggregate"]["total_claims"] == 1
    assert data["rows"][0]["summary"]["claims_total"] == 1


# -- review loop -------------------------------------------------------------
def test_adjudication_recomputes_metrics_without_mutating_original(tmp_path):
    batch = run_dataset([make_row("bad", CONTRADICTED_ANSWER)], MockJudge())
    qpath = tmp_path / "queue.jsonl"
    write_queue(batch, qpath)
    queue = [
        json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()
    ]
    assert len(queue) == 1
    entry = queue[0]
    assert entry["row_id"] == "bad"
    assert entry["claim_id"] == "c0"
    assert entry["verdict"] == "contradicted"
    assert entry["gold"] is None
    assert entry["spans"]
    assert all(set(s) == {"doc_id", "start", "end", "text"} for s in entry["spans"])

    entry["gold"] = "supported"
    adjudicated = ingest_adjudications(batch, queue)

    new_result = adjudicated.rows[0].report.results[0]
    assert new_result.verdict is Verdict.SUPPORTED
    assert new_result.confidence == 1.0
    assert new_result.rationale == "human adjudication"
    assert adjudicated.rows[0].report.contradicted_rate == 0.0
    assert adjudicated.rows[0].report.citation_precision == 1.0
    assert adjudicated.review_queue_size == 0

    original = batch.rows[0].report.results[0]
    assert original.verdict is Verdict.CONTRADICTED
    assert original.confidence != 1.0
    assert batch.review_queue_size == 1


def test_ingest_accepts_jsonl_path(tmp_path):
    batch = run_dataset([make_row("bad", CONTRADICTED_ANSWER)], MockJudge())
    queue = export_queue(batch)
    queue[0]["gold"] = "unsupported"
    path = tmp_path / "adjudicated.jsonl"
    write_jsonl(path, queue)
    adjudicated = ingest_adjudications(batch, path)
    assert adjudicated.rows[0].report.results[0].verdict is Verdict.UNSUPPORTED


def test_gold_null_rows_are_skipped_and_unknown_pairs_raise():
    batch = run_dataset([make_row("bad", CONTRADICTED_ANSWER)], MockJudge())
    same = ingest_adjudications(batch, export_queue(batch))  # gold still null
    assert same.to_dict() == batch.to_dict()

    with pytest.raises(ValueError, match="unknown"):
        ingest_adjudications(
            batch, [{"row_id": "nope", "claim_id": "c0", "gold": "supported"}]
        )


def test_to_fixture_rows_matches_fixture_schema():
    batch = run_dataset([make_row("bad", CONTRADICTED_ANSWER)], MockJudge())
    queue = export_queue(batch)
    queue[0]["gold"] = "contradicted"
    fixtures = to_fixture_rows(batch, queue)

    assert len(fixtures) == 1
    row = fixtures[0]
    assert set(row) == {"id", "claim", "spans", "gold", "tags"}
    assert row["id"] == "bad:c0"
    assert isinstance(row["claim"], str) and row["claim"]
    assert row["spans"] and all(isinstance(s, str) for s in row["spans"])
    assert row["gold"] in {v.value for v in Verdict}
    assert row["tags"] == ["adjudicated"]
    assert to_fixture_rows(batch, export_queue(batch)) == []  # gold still null
