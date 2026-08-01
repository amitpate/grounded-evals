"""Batch evaluation over JSONL datasets.

``run_dataset`` maps ``evaluate`` over every row of a dataset and rolls the
per-row ``EvalReport`` numbers up into a ``BatchReport``: per-metric means,
total review load, and abstention quality (did the generator refuse exactly
on the rows the corpus cannot answer). A row whose judge fails terminally
becomes an errored row instead of crashing the batch, so one bad generation
never costs a whole run.

Dataset format is JSONL, one object per line::

    {"id": str, "question": str, "answer": str,
     "docs": [{"doc_id": str, "title": str, "text": str}, ...],
     "answerable": bool,           # optional, default true
     "extra_markers": {str: str}}  # optional

``answerable: false`` marks a question the corpus cannot answer, where the
correct generator behavior is to abstain (``EvalReport.abstained``).

CLI::

    python -m grounded_evals.runner data.jsonl --out report.json
        [--review-queue queue.jsonl] [--judge mock|llm] [--workers N]
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .judge.base import Judge, JudgeError
from .pipeline import evaluate
from .schema import EvalReport, SourceDoc, _ratio

# EvalReport metrics averaged across rows. Each is Optional on a single row;
# None rows are excluded from the mean rather than counted as zero.
_MEAN_METRICS = (
    "citation_recall",
    "citation_precision",
    "per_citation_precision",
    "partial_rate",
    "contradicted_rate",
    "unsupported_rate",
    "ungrounded_rate",
    "review_load",
)

RowsOrPath = str | Path | Iterable[dict]


@dataclass(frozen=True)
class DatasetRow:
    """One validated dataset row.

    Attributes:
        row_id: Unique row identifier from the dataset.
        question: The question posed to the generator.
        answer: The generator's answer, citation markers included.
        docs: Source corpus the answer must be grounded in.
        answerable: Whether the corpus can answer the question. On
            unanswerable rows, abstaining is the correct behavior.
        extra_markers: Extra citation-marker resolutions passed through to
            ``evaluate``, or None.
    """

    row_id: str
    question: str
    answer: str
    docs: list[SourceDoc]
    answerable: bool
    extra_markers: dict[str, str] | None


@dataclass(frozen=True)
class RowResult:
    """Evaluation outcome for one dataset row.

    Attributes:
        row_id: The dataset row id.
        report: The per-row evaluation report.
        answerable: The row's answerable flag, kept for abstention scoring.
    """

    row_id: str
    report: EvalReport
    answerable: bool


@dataclass(frozen=True)
class RowError:
    """A row whose judge failed terminally.

    Attributes:
        row_id: The dataset row id.
        error: The ``JudgeError`` message.
    """

    row_id: str
    error: str


def _parse_row(obj: object, lineno: int) -> DatasetRow:
    """Validate one decoded dataset row.

    Args:
        obj: The decoded JSON value for this row.
        lineno: 1-based dataset line number, used in error messages.

    Returns:
        The validated ``DatasetRow``.

    Raises:
        ValueError: If the row does not match the dataset schema; the
            message names the offending line number.
    """

    def fail(msg: str) -> ValueError:
        return ValueError(f"dataset line {lineno}: {msg}")

    if not isinstance(obj, dict):
        raise fail("row must be a JSON object")
    for key in ("id", "question", "answer"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise fail(f'"{key}" must be a non-empty string')
    docs_raw = obj.get("docs")
    if not isinstance(docs_raw, list) or not docs_raw:
        raise fail('"docs" must be a non-empty list of source documents')
    docs: list[SourceDoc] = []
    seen_doc_ids: set[str] = set()
    for i, doc in enumerate(docs_raw):
        if not isinstance(doc, dict) or not all(
            isinstance(doc.get(k), str) for k in ("doc_id", "title", "text")
        ):
            raise fail(f'docs[{i}] must have string "doc_id", "title" and "text"')
        if doc["doc_id"] in seen_doc_ids:
            raise fail(f'duplicate doc_id {doc["doc_id"]!r} in "docs"')
        seen_doc_ids.add(doc["doc_id"])
        docs.append(SourceDoc(doc["doc_id"], doc["title"], doc["text"]))
    answerable = obj.get("answerable", True)
    if not isinstance(answerable, bool):
        raise fail('"answerable" must be a boolean')
    extra_markers = obj.get("extra_markers")
    if extra_markers is not None and (
        not isinstance(extra_markers, dict)
        or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra_markers.items()
        )
    ):
        raise fail('"extra_markers" must map marker strings to doc-id strings')
    return DatasetRow(
        obj["id"], obj["question"], obj["answer"], docs, answerable, extra_markers
    )


def _load_rows(rows_or_path: RowsOrPath) -> list[DatasetRow]:
    """Decode and validate dataset rows from a JSONL path or an iterable.

    Args:
        rows_or_path: Path to a JSONL file, or an iterable of row dicts
            (positions count as line numbers for error messages).

    Returns:
        Validated rows in dataset order.

    Raises:
        ValueError: On invalid JSON, a schema violation, or a duplicate row
            id; the message names the offending line number.
    """
    numbered: list[tuple[object, int]] = []
    if isinstance(rows_or_path, (str, Path)):
        with open(rows_or_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    numbered.append((json.loads(line), lineno))
                except json.JSONDecodeError as e:
                    raise ValueError(f"dataset line {lineno}: invalid JSON: {e}") from e
    else:
        numbered = [(obj, lineno) for lineno, obj in enumerate(rows_or_path, start=1)]

    rows: list[DatasetRow] = []
    seen: set[str] = set()
    for obj, lineno in numbered:
        row = _parse_row(obj, lineno)
        if row.row_id in seen:
            raise ValueError(f"dataset line {lineno}: duplicate row id {row.row_id!r}")
        seen.add(row.row_id)
        rows.append(row)
    return rows


@dataclass
class BatchReport:
    """Aggregated evaluation results for one dataset run.

    Attributes:
        rows: Successful per-row results, in dataset order.
        errors: Rows whose judge failed terminally, in dataset order.
        judge_fingerprint: Judge identity for reproducibility, or None when
            the judge does not expose one.
    """

    rows: list[RowResult]
    errors: list[RowError] = field(default_factory=list)
    judge_fingerprint: dict | None = None

    @property
    def total_claims(self) -> int:
        """Total claims extracted across all evaluated rows."""
        return sum(len(r.report.results) for r in self.rows)

    @property
    def review_queue_size(self) -> int:
        """Total claims routed to human review across all evaluated rows."""
        return sum(len(r.report.review_queue) for r in self.rows)

    @property
    def review_load(self) -> float | None:
        """Fraction of all claims routed to review; None with no claims."""
        return _ratio(self.review_queue_size, self.total_claims)

    @property
    def mean_metrics(self) -> dict[str, float | None]:
        """Mean of each per-row summary metric, skipping rows where it is None.

        A row with nothing to measure (e.g. no cited claims) does not drag a
        mean toward zero; a metric that is None on every row stays None.

        Returns:
            Mapping from ``mean_<metric>`` to the mean, in a fixed order.
        """
        means: dict[str, float | None] = {}
        for name in _MEAN_METRICS:
            values = [
                value
                for r in self.rows
                if (value := getattr(r.report, name)) is not None
            ]
            means[f"mean_{name}"] = sum(values) / len(values) if values else None
        return means

    @property
    def correct_refusal_rate(self) -> float | None:
        """Of unanswerable rows, fraction where the answer abstained.

        None when the dataset has no ``answerable: false`` rows.
        """
        unanswerable = [r for r in self.rows if not r.answerable]
        return _ratio(
            sum(1 for r in unanswerable if r.report.abstained), len(unanswerable)
        )

    @property
    def false_refusal_rate(self) -> float | None:
        """Of answerable rows, fraction where the answer abstained anyway.

        None when the dataset has no answerable rows.
        """
        answerable = [r for r in self.rows if r.answerable]
        return _ratio(sum(1 for r in answerable if r.report.abstained), len(answerable))

    def summary(self) -> dict:
        """Aggregate numbers only, rounded for printing.

        Returns:
            Row counts, claim and review totals, abstention quality, and the
            per-metric means.
        """

        def rnd(x: float | None) -> float | None:
            return None if x is None else round(x, 3)

        return {
            "rows_evaluated": len(self.rows),
            "rows_errored": len(self.errors),
            "total_claims": self.total_claims,
            "review_queue_size": self.review_queue_size,
            "review_load": rnd(self.review_load),
            "correct_refusal_rate": rnd(self.correct_refusal_rate),
            "false_refusal_rate": rnd(self.false_refusal_rate),
            **{name: rnd(value) for name, value in self.mean_metrics.items()},
        }

    def to_dict(self) -> dict:
        """Self-contained JSON-ready report.

        Returns:
            The package version, judge fingerprint, dataset row count,
            aggregate summary, per-row summaries, and errored rows -- enough
            to reproduce and audit the run without the Python objects.
        """
        return {
            "grounded_evals_version": __version__,
            "judge_fingerprint": self.judge_fingerprint,
            "dataset_rows": len(self.rows) + len(self.errors),
            "aggregate": self.summary(),
            "rows": [
                {
                    "id": r.row_id,
                    "answerable": r.answerable,
                    "summary": r.report.summary(),
                }
                for r in self.rows
            ],
            "errors": [{"id": e.row_id, "error": e.error} for e in self.errors],
        }

    def save(self, path: str | Path) -> None:
        """Write ``to_dict()`` as indented JSON to ``path``.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


def run_dataset(
    rows_or_path: RowsOrPath,
    judge: Judge,
    *,
    max_workers: int = 1,
) -> BatchReport:
    """Evaluate every dataset row and aggregate the reports.

    Args:
        rows_or_path: Path to a JSONL dataset file, or an iterable of
            already-decoded row dicts (see the module docstring for the row
            schema).
        judge: Judge used for claim extraction and verdicts.
        max_workers: Number of evaluation threads. Results are kept in
            dataset order regardless of scheduling, so parallel runs are
            identical to serial ones.

    Returns:
        A ``BatchReport`` with one ``RowResult`` per evaluated row and one
        ``RowError`` per row whose judge raised ``JudgeError``; a failing
        row never crashes the batch.

    Raises:
        ValueError: If a dataset row is malformed; the message names the
            offending line number.
    """
    rows = _load_rows(rows_or_path)

    def run(row: DatasetRow) -> EvalReport:
        return evaluate(
            row.question, row.answer, row.docs, judge, extra_markers=row.extra_markers
        )

    outcomes: list[EvalReport | JudgeError] = []
    if max_workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run, row) for row in rows]
            for future in futures:
                try:
                    outcomes.append(future.result())
                except JudgeError as e:
                    outcomes.append(e)
    else:
        for row in rows:
            try:
                outcomes.append(run(row))
            except JudgeError as e:
                outcomes.append(e)

    results: list[RowResult] = []
    errors: list[RowError] = []
    for row, outcome in zip(rows, outcomes, strict=True):
        if isinstance(outcome, JudgeError):
            errors.append(RowError(row.row_id, str(outcome)))
        else:
            results.append(RowResult(row.row_id, outcome, row.answerable))
    return BatchReport(
        rows=results,
        errors=errors,
        judge_fingerprint=getattr(judge, "fingerprint", None),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see the module docstring for usage.

    Args:
        argv: Argument list, or None to use ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="python -m grounded_evals.runner",
        description="Batch-evaluate a JSONL dataset of grounded answers.",
    )
    parser.add_argument("dataset", help="path to the dataset JSONL file")
    parser.add_argument("--out", required=True, help="path for the JSON batch report")
    parser.add_argument(
        "--review-queue", help="optional path for the human-review queue JSONL"
    )
    parser.add_argument(
        "--judge",
        choices=("mock", "llm"),
        default="mock",
        help="mock is keyless and deterministic; llm reads GROUNDED_EVALS_* env vars",
    )
    parser.add_argument("--workers", type=int, default=1, help="evaluation threads")
    args = parser.parse_args(argv)

    judge: Judge
    if args.judge == "llm":
        from .judge.llm import LLMJudge

        judge = LLMJudge.from_env()
    else:
        from .judge.mock import MockJudge

        judge = MockJudge()

    batch = run_dataset(args.dataset, judge, max_workers=args.workers)
    batch.save(args.out)
    if args.review_queue:
        from .review import write_queue  # local import: review imports this module

        write_queue(batch, args.review_queue)
    print(json.dumps(batch.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
