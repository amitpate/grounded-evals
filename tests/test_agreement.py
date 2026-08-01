from pathlib import Path

import pytest

from grounded_evals.agreement import GOLD_VERDICTS, load_fixtures, measure
from grounded_evals.judge import MockJudge

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "labelled_claims.jsonl"


@pytest.fixture(scope="module")
def fixtures():
    return load_fixtures(FIXTURES_PATH)


@pytest.fixture(scope="module")
def report(fixtures):
    return measure(MockJudge(), fixtures)


def test_fixtures_load_and_validate(fixtures):
    # load_fixtures raises on any malformed row, so loading is the validation.
    assert 50 <= len(fixtures) <= 70
    assert len({f.id for f in fixtures}) == len(fixtures)
    for f in fixtures:
        assert f.claim.strip()
        assert f.spans and all(s.strip() for s in f.spans)
        assert f.tags


def test_gold_values_are_only_the_four_allowed(fixtures):
    allowed = set(GOLD_VERDICTS)
    assert {f.gold for f in fixtures} == allowed  # every class represented, no others


def test_mock_judge_beats_the_agreement_floor(report):
    # The mock is the floor. If it cannot beat 40% on the fixtures, either the
    # fixtures or the mock are broken.
    assert report.accuracy > 0.4


def test_confusion_matrix_counts_sum_to_fixture_count(fixtures, report):
    total = sum(n for row in report.confusion.values() for n in row.values())
    assert total == len(fixtures)


def test_ece_is_a_probability_gap(report):
    assert 0.0 <= report.ece <= 1.0


def test_recommend_threshold_unreachable_target_is_none(report):
    assert report.recommend_threshold(1.01) is None


def test_recommend_threshold_modest_target_is_a_float(report):
    threshold = report.recommend_threshold(0.5)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_sweep_coverage_is_monotonically_non_increasing(report):
    coverages = [p.coverage for p in report.sweep]
    assert all(a >= b for a, b in zip(coverages, coverages[1:], strict=False))


def test_malformed_row_names_its_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = (
        '{"id": "ok-1", "claim": "The sky is blue.", "spans": ["The sky is blue."], '
        '"gold": "supported", "tags": ["paraphrase"]}'
    )
    path.write_text(good + "\n" + '{"id": "bad-1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        load_fixtures(path)


def test_bad_gold_label_is_rejected(tmp_path):
    path = tmp_path / "bad_gold.jsonl"
    path.write_text(
        '{"id": "x", "claim": "Water is wet.", "spans": ["Water is wet."], '
        '"gold": "not_checkable", "tags": ["unrelated"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row 1"):
        load_fixtures(path)


def test_measure_rejects_empty_fixture_set():
    with pytest.raises(ValueError):
        measure(MockJudge(), [])
