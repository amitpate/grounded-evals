"""Judge-agreement and calibration measurement.

A judge only earns automation to the extent it agrees with human labels, so
this module runs any ``Judge`` against a labelled fixture set (claim + source
spans + gold verdict) and reports where it agrees, where it fails, and how
honest its confidence is. Running ``MockJudge`` over
``fixtures/labelled_claims.jsonl`` produces the *judge agreement floor*: the
score a model-backed judge must beat before its verdicts are worth anything.

Calibration follows the standard recipe: confidence bucketed into 10 bins,
expected calibration error (ECE) as the count-weighted gap between each bin's
mean confidence and its accuracy. The threshold sweep supports selective
evaluation (Trust-or-Escalate style): pick the smallest confidence threshold
whose auto-accepted verdicts hit a target accuracy and route everything below
it to human review.

Run standalone::

    python -m grounded_evals.agreement [fixtures_path]
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .judge.base import Judge
from .judge.mock import MockJudge
from .schema import Claim, SourceSpan, Verdict

#: Verdicts a fixture may be labelled with. NOT_CHECKABLE is deliberately
#: excluded: fixtures are checkable claims by construction, and a judge that
#: predicts it here is simply wrong.
GOLD_VERDICTS: tuple[Verdict, ...] = (
    Verdict.SUPPORTED,
    Verdict.PARTIAL,
    Verdict.CONTRADICTED,
    Verdict.UNSUPPORTED,
)

CALIBRATION_BINS = 10
SWEEP_STEP = 0.05
DEFAULT_FIXTURES_PATH = Path("fixtures") / "labelled_claims.jsonl"

_FIXTURE_DOC_ID = "fixture"
_REQUIRED_KEYS = frozenset({"id", "claim", "spans", "gold", "tags"})


def _ratio(num: int, denom: int) -> float | None:
    """None when there is nothing to measure -- never a fake 0.0."""
    return num / denom if denom else None


@dataclass(frozen=True)
class Fixture:
    """One labelled row: a claim, the spans to judge it against, and gold.

    Attributes:
        id: Unique row identifier.
        claim: Self-contained claim text.
        spans: Source span texts the claim is judged against.
        gold: Human-labelled verdict; one of ``GOLD_VERDICTS``.
        tags: Failure-mode tags ("numeric-swap", "negation", ...).
    """

    id: str
    claim: str
    spans: tuple[str, ...]
    gold: Verdict
    tags: tuple[str, ...]


def _str_items(value: object, lineno: int, key: str, *, required: bool) -> tuple[str, ...]:
    """Validate a JSON list of non-empty strings; return it as a tuple."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"row {lineno}: {key!r} must be a list of non-empty strings")
    if required and not value:
        raise ValueError(f"row {lineno}: {key!r} must not be empty")
    return tuple(value)


def _parse_row(row: object, lineno: int) -> Fixture:
    """Validate one decoded JSONL row and build a Fixture from it."""
    if not isinstance(row, dict):
        raise ValueError(f"row {lineno}: expected a JSON object")
    if set(row) != _REQUIRED_KEYS:
        missing = _REQUIRED_KEYS - set(row)
        unknown = set(row) - _REQUIRED_KEYS
        raise ValueError(
            f"row {lineno}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    fixture_id, claim, gold = row["id"], row["claim"], row["gold"]
    if not isinstance(fixture_id, str) or not fixture_id.strip():
        raise ValueError(f"row {lineno}: 'id' must be a non-empty string")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"row {lineno}: 'claim' must be a non-empty string")
    allowed = {v.value for v in GOLD_VERDICTS}
    if gold not in allowed:
        raise ValueError(f"row {lineno}: 'gold' must be one of {sorted(allowed)}, got {gold!r}")
    return Fixture(
        id=fixture_id,
        claim=claim,
        spans=_str_items(row["spans"], lineno, "spans", required=True),
        gold=Verdict(gold),
        tags=_str_items(row["tags"], lineno, "tags", required=False),
    )


def load_fixtures(path: str | Path) -> list[Fixture]:
    """Load and validate a labelled fixture set.

    Args:
        path: JSONL file; each non-blank line is an object with keys
            ``id``, ``claim``, ``spans``, ``gold``, ``tags``.

    Returns:
        The validated fixtures, in file order.

    Raises:
        ValueError: On the first malformed or duplicate-id row, naming its
            1-based line number.
    """
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {lineno}: invalid JSON: {exc}") from exc
        fixture = _parse_row(row, lineno)
        if fixture.id in seen:
            raise ValueError(f"row {lineno}: duplicate id {fixture.id!r}")
        seen.add(fixture.id)
        fixtures.append(fixture)
    return fixtures


@dataclass(frozen=True)
class Outcome:
    """One fixture's judged result, kept for drill-down and error review."""

    fixture: Fixture
    predicted: Verdict
    confidence: float

    @property
    def correct(self) -> bool:
        return self.predicted is self.fixture.gold


@dataclass(frozen=True)
class ClassMetrics:
    """Precision/recall/F1 for one gold class; None when undefined."""

    precision: float | None
    recall: float | None
    f1: float | None
    support: int


@dataclass(frozen=True)
class TagAccuracy:
    """Agreement over the fixtures carrying one failure-mode tag."""

    count: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.count


@dataclass(frozen=True)
class CalibrationBin:
    """One confidence bucket: [lower, upper) except the last, which is closed."""

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None


@dataclass(frozen=True)
class ThresholdPoint:
    """Selective-evaluation operating point at one confidence threshold.

    Attributes:
        threshold: Minimum confidence to auto-accept a verdict.
        coverage: Fraction of verdicts with confidence >= threshold.
        accuracy: Agreement among auto-accepted verdicts; None when the
            threshold auto-accepts nothing.
    """

    threshold: float
    coverage: float
    accuracy: float | None


def _fixture_spans(fixture: Fixture) -> list[SourceSpan]:
    """Span texts as SourceSpans under a synthetic doc with running offsets."""
    spans: list[SourceSpan] = []
    cursor = 0
    for text in fixture.spans:
        spans.append(SourceSpan(_FIXTURE_DOC_ID, cursor, cursor + len(text), text))
        cursor += len(text) + 1  # +1 for an implicit separator between spans
    return spans


def _class_metrics(outcomes: list[Outcome], cls: Verdict) -> ClassMetrics:
    tp = sum(1 for o in outcomes if o.fixture.gold is cls and o.predicted is cls)
    predicted = sum(1 for o in outcomes if o.predicted is cls)
    support = sum(1 for o in outcomes if o.fixture.gold is cls)
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, support)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return ClassMetrics(precision, recall, f1, support)


def _confusion(outcomes: list[Outcome]) -> dict[str, dict[str, int]]:
    """Dense over the four gold classes; extra predicted labels appear as
    columns only when observed, so counts always sum to len(outcomes)."""
    matrix: dict[str, dict[str, int]] = {
        gold.value: {pred.value: 0 for pred in GOLD_VERDICTS} for gold in GOLD_VERDICTS
    }
    for o in outcomes:
        row = matrix[o.fixture.gold.value]
        row[o.predicted.value] = row.get(o.predicted.value, 0) + 1
    return matrix


def _per_tag(outcomes: list[Outcome]) -> dict[str, TagAccuracy]:
    counts: dict[str, list[int]] = {}
    for o in outcomes:
        for tag in o.fixture.tags:
            count_correct = counts.setdefault(tag, [0, 0])
            count_correct[0] += 1
            count_correct[1] += o.correct
    return {
        tag: TagAccuracy(count=n, correct=c) for tag, (n, c) in sorted(counts.items())
    }


def _calibration(outcomes: list[Outcome]) -> tuple[list[CalibrationBin], float]:
    """10-bin reliability data and expected calibration error."""
    binned: list[list[Outcome]] = [[] for _ in range(CALIBRATION_BINS)]
    for o in outcomes:
        index = min(int(o.confidence * CALIBRATION_BINS), CALIBRATION_BINS - 1)
        binned[index].append(o)
    bins: list[CalibrationBin] = []
    ece = 0.0
    for i, members in enumerate(binned):
        lower, upper = i / CALIBRATION_BINS, (i + 1) / CALIBRATION_BINS
        if not members:
            bins.append(CalibrationBin(lower, upper, 0, None, None))
            continue
        mean_conf = sum(o.confidence for o in members) / len(members)
        accuracy = sum(o.correct for o in members) / len(members)
        bins.append(CalibrationBin(lower, upper, len(members), mean_conf, accuracy))
        ece += len(members) / len(outcomes) * abs(accuracy - mean_conf)
    return bins, ece


def _sweep(outcomes: list[Outcome]) -> list[ThresholdPoint]:
    points: list[ThresholdPoint] = []
    steps = int(round(1.0 / SWEEP_STEP)) + 1
    for i in range(steps):
        threshold = round(i * SWEEP_STEP, 2)
        accepted = [o for o in outcomes if o.confidence >= threshold]
        points.append(
            ThresholdPoint(
                threshold=threshold,
                coverage=len(accepted) / len(outcomes),
                accuracy=_ratio(sum(o.correct for o in accepted), len(accepted)),
            )
        )
    return points


@dataclass
class AgreementReport:
    """Everything ``measure`` learned about one judge on one fixture set."""

    outcomes: list[Outcome]
    accuracy: float
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]
    per_tag: dict[str, TagAccuracy]
    calibration: list[CalibrationBin]
    ece: float
    sweep: list[ThresholdPoint]

    def recommend_threshold(self, target_accuracy: float) -> float | None:
        """Smallest threshold whose auto-accepted accuracy meets the target.

        Selective evaluation: verdicts at or above the returned confidence
        threshold are auto-accepted, the rest go to human review. Lower is
        better (more coverage, less review load).

        Args:
            target_accuracy: Required accuracy among auto-accepted verdicts.

        Returns:
            The smallest qualifying threshold, or None when no threshold that
            still accepts anything reaches the target.
        """
        for point in self.sweep:
            if point.accuracy is not None and point.accuracy >= target_accuracy:
                return point.threshold
        return None

    def summary(self) -> dict:
        """JSON-serializable digest of the report."""

        def rnd(x: float | None) -> float | None:
            return None if x is None else round(x, 3)

        return {
            "fixtures": len(self.outcomes),
            "accuracy": rnd(self.accuracy),
            "per_class": {
                cls: {
                    "precision": rnd(m.precision),
                    "recall": rnd(m.recall),
                    "f1": rnd(m.f1),
                    "support": m.support,
                }
                for cls, m in self.per_class.items()
            },
            "confusion": self.confusion,
            "per_tag": {
                tag: {"count": t.count, "accuracy": rnd(t.accuracy)}
                for tag, t in self.per_tag.items()
            },
            "ece": rnd(self.ece),
            "calibration": [
                {
                    "range": [b.lower, b.upper],
                    "count": b.count,
                    "mean_confidence": rnd(b.mean_confidence),
                    "accuracy": rnd(b.accuracy),
                }
                for b in self.calibration
            ],
            "threshold_sweep": [
                {
                    "threshold": p.threshold,
                    "coverage": rnd(p.coverage),
                    "accuracy": rnd(p.accuracy),
                }
                for p in self.sweep
            ],
        }

    def format(self) -> str:
        """Readable multi-line report for terminal output."""

        def pct(x: float | None) -> str:
            return "  -- " if x is None else f"{x:5.1%}"

        lines = [
            f"fixtures: {len(self.outcomes)}   "
            f"overall agreement: {self.accuracy:.1%}   ECE: {self.ece:.3f}",
            "",
            "per-class            prec   recall  f1     support",
        ]
        for cls, m in self.per_class.items():
            lines.append(
                f"  {cls:<18} {pct(m.precision)}  {pct(m.recall)}  {pct(m.f1)}  {m.support:>4}"
            )
        lines.append("")
        lines.append("confusion (gold -> predicted)")
        for gold, row in self.confusion.items():
            cells = "  ".join(f"{pred}={n}" for pred, n in row.items() if n)
            lines.append(f"  {gold:<18} {cells or '-'}")
        lines.append("")
        lines.append("per-tag agreement")
        for tag, t in self.per_tag.items():
            lines.append(f"  {tag:<18} {pct(t.accuracy)}  ({t.correct}/{t.count})")
        lines.append("")
        lines.append("threshold sweep (confidence >= t auto-accepted)")
        for p in self.sweep:
            lines.append(
                f"  t={p.threshold:.2f}  coverage {pct(p.coverage)}  accuracy {pct(p.accuracy)}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def measure(judge: Judge, fixtures: Sequence[Fixture]) -> AgreementReport:
    """Run a judge over labelled fixtures and score its agreement.

    Each fixture becomes one ``judge.verdict`` call: a Claim built from the
    fixture's claim text and SourceSpans built from its span texts under the
    synthetic doc id "fixture" with running char offsets.

    Args:
        judge: Any Judge implementation.
        fixtures: Labelled fixtures, e.g. from ``load_fixtures``.

    Returns:
        An AgreementReport with accuracy, per-class and per-tag breakdowns,
        the confusion matrix, calibration data, and the threshold sweep.

    Raises:
        ValueError: If ``fixtures`` is empty -- there is nothing to measure.
    """
    if not fixtures:
        raise ValueError("cannot measure agreement on an empty fixture set")

    outcomes: list[Outcome] = []
    for fixture in fixtures:
        claim = Claim(claim_id=fixture.id, text=fixture.claim)
        jv = judge.verdict(claim, _fixture_spans(fixture))
        outcomes.append(Outcome(fixture, jv.verdict, jv.confidence))

    calibration, ece = _calibration(outcomes)
    return AgreementReport(
        outcomes=outcomes,
        accuracy=sum(o.correct for o in outcomes) / len(outcomes),
        per_class={cls.value: _class_metrics(outcomes, cls) for cls in GOLD_VERDICTS},
        confusion=_confusion(outcomes),
        per_tag=_per_tag(outcomes),
        calibration=calibration,
        ece=ece,
        sweep=_sweep(outcomes),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: MockJudge vs the fixtures = the agreement floor."""
    parser = argparse.ArgumentParser(
        prog="python -m grounded_evals.agreement",
        description=(
            "Measure MockJudge agreement against a labelled fixture set. "
            "The result is the floor a model-backed judge must beat."
        ),
    )
    parser.add_argument(
        "fixtures_path",
        nargs="?",
        default=str(DEFAULT_FIXTURES_PATH),
        help=f"labelled JSONL fixtures (default: {DEFAULT_FIXTURES_PATH})",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.9,
        help="target auto-accept accuracy for the threshold recommendation",
    )
    args = parser.parse_args(argv)

    report = measure(MockJudge(), load_fixtures(args.fixtures_path))
    print(report.format())
    recommended = report.recommend_threshold(args.target_accuracy)
    print()
    if recommended is None:
        print(
            f"no confidence threshold reaches {args.target_accuracy:.0%} "
            "auto-accept accuracy: route everything to review"
        )
    else:
        print(
            f"recommended threshold for {args.target_accuracy:.0%} "
            f"auto-accept accuracy: {recommended:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
