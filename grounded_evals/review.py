"""Human-review loop: export low-confidence verdicts, ingest adjudications.

Closes the flywheel around the automated judge (the FaithJudge pattern):

1. ``export_queue`` / ``write_queue`` turn every ``ClaimResult`` that needs
   review into an annotation row with an empty ``gold`` field.
2. A human fills ``gold`` with the corrected verdict.
3. ``ingest_adjudications`` rebuilds the batch with the human verdicts
   swapped in at confidence 1.0, so every aggregate metric recomputes
   automatically -- the original batch is never mutated.
4. ``to_fixture_rows`` converts adjudicated claims into agreement fixtures,
   so human decisions keep re-anchoring judge calibration.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from pathlib import Path

from .runner import BatchReport
from .schema import ClaimResult, Verdict

Adjudications = str | Path | Iterable[dict]

# Annotators may answer with any verdict, including "not_checkable" ("this
# is not a factual claim at all") — ingestion flips the claim's checkable
# flag for that case so it leaves every metric denominator.
_GOLD_VALUES = frozenset(v.value for v in Verdict)


def export_queue(batch: BatchReport) -> list[dict]:
    """Collect every claim needing review as an annotation row.

    Args:
        batch: A batch report produced by ``runner.run_dataset``.

    Returns:
        One dict per ``ClaimResult`` in any row's review queue, in dataset
        order. ``spans`` carries the alignment spans the verdict was judged
        against, so the annotator sees exactly the evidence the judge saw.
        ``gold`` starts as None; the annotator fills it with the corrected
        verdict (one of the ``Verdict`` values).
    """
    queue: list[dict] = []
    for row in batch.rows:
        for r in row.report.review_queue:
            queue.append(
                {
                    "row_id": row.row_id,
                    "claim_id": r.claim.claim_id,
                    "claim": r.claim.text,
                    "citations": [c.doc_id for c in r.claim.citations],
                    "verdict": r.verdict.value,
                    "confidence": r.confidence,
                    "rationale": r.rationale,
                    "spans": [
                        {"doc_id": s.doc_id, "start": s.start, "end": s.end, "text": s.text}
                        for s in r.alignment.spans
                    ],
                    "gold": None,
                }
            )
    return queue


def write_queue(batch: BatchReport, path: str | Path) -> None:
    """Write the review queue as JSONL, one annotation row per line.

    Args:
        batch: A batch report produced by ``runner.run_dataset``.
        path: Destination file path.
    """
    lines = [json.dumps(row) for row in export_queue(batch)]
    Path(path).write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def _load_adjudications(adjudications: Adjudications) -> list[dict]:
    """Read adjudication rows from a JSONL path or pass an iterable through.

    Args:
        adjudications: Path to a JSONL file, or an iterable of dicts.

    Returns:
        The decoded adjudication rows, in input order.

    Raises:
        ValueError: If a JSONL line is not valid JSON; the message names the
            offending line number.
    """
    if isinstance(adjudications, (str, Path)):
        rows: list[dict] = []
        with open(adjudications, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"adjudication line {lineno}: invalid JSON: {e}"
                    ) from e
        return rows
    return list(adjudications)


def _claims_by_key(batch: BatchReport) -> dict[tuple[str, str], ClaimResult]:
    """Index every claim result in the batch by (row_id, claim_id)."""
    return {
        (row.row_id, result.claim.claim_id): result
        for row in batch.rows
        for result in row.report.results
    }


def _gold_map(
    batch: BatchReport, adjudications: Adjudications
) -> dict[tuple[str, str], Verdict]:
    """Resolve filled adjudications to (row_id, claim_id) -> gold verdict.

    Rows whose ``gold`` is still None are skipped.

    Args:
        batch: The batch the adjudications refer to.
        adjudications: Queue rows with ``gold`` filled (path or iterable).

    Returns:
        Mapping from (row_id, claim_id) to the human verdict, in
        adjudication order.

    Raises:
        ValueError: If an adjudication references an unknown
            (row_id, claim_id) pair, or its ``gold`` is not a valid verdict.
    """
    known = _claims_by_key(batch)
    gold: dict[tuple[str, str], Verdict] = {}
    for adj in _load_adjudications(adjudications):
        if adj.get("gold") is None:
            continue
        key = (adj.get("row_id"), adj.get("claim_id"))
        if key not in known:
            raise ValueError(
                f"adjudication references unknown claim {key[0]!r}/{key[1]!r}"
            )
        value = adj["gold"]
        if value not in _GOLD_VALUES:
            raise ValueError(
                f'"gold" must be one of {sorted(_GOLD_VALUES)}, got {value!r}'
            )
        gold[key] = Verdict(value)
    return gold


def ingest_adjudications(
    batch: BatchReport, adjudications: Adjudications
) -> BatchReport:
    """Fold human adjudications back into a batch report.

    Args:
        batch: The batch the adjudications refer to; it is not mutated.
        adjudications: Review-queue rows with ``gold`` filled, as an iterable
            of dicts or a JSONL path. Rows with ``gold`` still None are
            skipped.

    Returns:
        A new, fully independent ``BatchReport`` (deep copy — no object is
        shared with the input) in which every adjudicated ``ClaimResult``
        carries the human verdict at confidence 1.0 with rationale
        "human adjudication". A gold of ``not_checkable`` additionally flips
        the claim's ``checkable`` flag, removing it from metric denominators.
        All aggregate metrics recompute from the new verdicts automatically.

    Raises:
        ValueError: If an adjudication references an unknown
            (row_id, claim_id) pair, or its ``gold`` is not a valid verdict.
    """
    gold = _gold_map(batch, adjudications)
    adjudicated = copy.deepcopy(batch)
    for row in adjudicated.rows:
        for result in row.report.results:
            verdict = gold.get((row.row_id, result.claim.claim_id))
            if verdict is None:
                continue
            result.verdict = verdict
            result.confidence = 1.0
            result.rationale = "human adjudication"
            if verdict is Verdict.NOT_CHECKABLE:
                result.claim.checkable = False
    return adjudicated


def to_fixture_rows(batch: BatchReport, adjudications: Adjudications) -> list[dict]:
    """Convert adjudicated claims into agreement-fixture rows.

    Each human decision becomes a labelled fixture the judge can be scored
    against later, so the labelled set grows with every review pass.

    Args:
        batch: The batch the adjudications refer to.
        adjudications: Review-queue rows with ``gold`` filled (path or
            iterable). Rows with ``gold`` still None are skipped.

    Returns:
        Fixture rows of the form ``{"id": "<row_id>:<claim_id>",
        "claim": str, "spans": [span texts], "gold": str,
        "tags": ["adjudicated"]}``, in adjudication order. Claims adjudicated
        as ``not_checkable`` are skipped — the agreement fixture set measures
        verdict quality on checkable claims only, and must stay loadable by
        ``agreement.load_fixtures``.

    Raises:
        ValueError: If an adjudication references an unknown
            (row_id, claim_id) pair, or its ``gold`` is not a valid verdict.
    """
    gold = _gold_map(batch, adjudications)
    known = _claims_by_key(batch)
    return [
        {
            "id": f"{row_id}:{claim_id}",
            "claim": known[(row_id, claim_id)].claim.text,
            "spans": [s.text for s in known[(row_id, claim_id)].alignment.spans],
            "gold": verdict.value,
            "tags": ["adjudicated"],
        }
        for (row_id, claim_id), verdict in gold.items()
        if verdict is not Verdict.NOT_CHECKABLE
    ]
