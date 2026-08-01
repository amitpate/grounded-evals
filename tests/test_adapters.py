"""Benchmark-adapter tests on synthetic records shaped like the real files.

No real benchmark data is vendored; these records mimic the published field
layouts (RAGTruth source_info/response JSONL, ALCE result JSON) so the
transformations stay honest without shipping third-party text.
"""
import pytest

from grounded_evals.adapters import (
    alce_dataset_rows,
    ragtruth_dataset_rows,
    ragtruth_fixture_rows,
)
from grounded_evals.agreement import load_fixtures
from grounded_evals.judge import MockJudge
from grounded_evals.runner import run_dataset

RT_SOURCES = [
    {
        "source_id": "s1",
        "task_type": "QA",
        "source_info": {
            "question": "When did the stadium reopen?",
            "passages": [
                "The stadium seats 90,000 people and reopened in 2007.",
                "The arch spans 315 metres across the stadium.",
            ],
        },
    },
    {
        "source_id": "s2",
        "task_type": "QA",
        "source_info": {
            "question": "What is the capacity?",
            "passages": (
                "passage 1: The stadium seats 90,000 people and reopened in 2007. "
                "passage 2: The arch spans 315 metres across the stadium."
            ),
        },
    },
    {"source_id": "s3", "task_type": "Summary", "source_info": "irrelevant"},
]

RT_RESPONSES = [
    {
        "id": "r1",
        "source_id": "s1",
        "model": "some-model",
        "response": "The stadium reopened in 2010.",
        "labels": [
            {
                "label_type": "Evident Conflict",
                "start": 0,
                "end": 29,
                "text": "The stadium reopened in 2010.",
            }
        ],
    },
    {
        "id": "r2",
        "source_id": "s2",
        "model": "some-model",
        "response": "The stadium seats 90,000 people.",
        "labels": [],
    },
]


def test_ragtruth_dataset_rows_run_end_to_end():
    rows = ragtruth_dataset_rows(RT_SOURCES, RT_RESPONSES)
    assert [r["id"] for r in rows] == ["s1:r1", "s2:r2"]  # Summary row filtered
    # Both passage encodings (list and "passage N:" string) yield two docs.
    assert all(len(r["docs"]) == 2 for r in rows)

    batch = run_dataset(rows, MockJudge())
    assert len(batch.rows) == 2
    # The 2010-vs-2007 response is caught as contradicted by the pipeline.
    assert batch.rows[0].report.contradicted_rate == 1.0


def test_ragtruth_fixture_rows_load_through_agreement(tmp_path):
    fixtures = ragtruth_fixture_rows(RT_SOURCES, RT_RESPONSES)
    assert len(fixtures) == 1
    assert fixtures[0]["gold"] == "contradicted"  # Evident Conflict mapped
    assert "ragtruth" in fixtures[0]["tags"]

    # The exported rows must load through the agreement module's validator.
    import json

    path = tmp_path / "rt_fixtures.jsonl"
    path.write_text("\n".join(json.dumps(f) for f in fixtures) + "\n")
    loaded = load_fixtures(path)
    assert loaded[0].gold.value == "contradicted"


def test_ragtruth_unknown_label_type_fails_loudly():
    responses = [
        {
            "id": "r9",
            "source_id": "s1",
            "response": "x" * 20,
            "labels": [{"label_type": "Brand New Category", "text": "some span"}],
        }
    ]
    with pytest.raises(ValueError, match="unknown label_type"):
        ragtruth_fixture_rows(RT_SOURCES, responses)


def test_ragtruth_unknown_source_fails_loudly():
    with pytest.raises(ValueError, match="unknown source"):
        ragtruth_dataset_rows(RT_SOURCES, [{"id": "r0", "source_id": "nope", "response": "x"}])


def test_alce_rows_exercise_marker_resolution():
    records = [
        {
            "question": "Compare the stadiums.",
            "output": (
                "The stadium seats 90,000 people and reopened in 2007. [1] "
                "The arch spans 315 metres. [2]"
            ),
            "docs": [
                {"title": "A", "text": "The stadium seats 90,000 people and reopened in 2007."},
                {"title": "B", "text": "The arch spans 315 metres across the stadium."},
            ],
        }
    ]
    rows = alce_dataset_rows(records)
    assert [d["doc_id"] for d in rows[0]["docs"]] == ["1", "2"]

    batch = run_dataset(rows, MockJudge())
    report = batch.rows[0].report
    # Both [N] markers resolved to their positional docs and both claims hold.
    assert report.citation_recall == 1.0
    assert report.citation_precision == 1.0
    assert report.per_citation_precision == 1.0


def test_alce_missing_fields_fail_loudly():
    with pytest.raises(ValueError, match="alce record 1"):
        alce_dataset_rows([{"question": "q", "docs": []}])
