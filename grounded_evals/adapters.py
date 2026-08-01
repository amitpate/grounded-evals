"""Adapters from public grounding benchmarks to grounded-evals formats.

External datasets are not vendored — download them yourself, then convert:

- **RAGTruth** (Niu et al., ACL 2024): span-level human hallucination labels
  on RAG responses. ``ragtruth_dataset_rows`` turns QA records into runner
  dataset rows; ``ragtruth_fixture_rows`` turns the human span labels into
  agreement fixtures, mapping RAGTruth's taxonomy onto ours — *conflict*
  labels become ``contradicted``, *baseless* labels become ``unsupported``.
  Running the fixture converter grows the labelled set with externally
  annotated ground truth, the same way review adjudications do.
- **ALCE** (Gao et al., EMNLP 2023): citation-generation benchmark whose
  result files carry answers with ``[1]``-style markers over a per-question
  doc list — exactly the marker convention ``pipeline.evaluate`` resolves.

Both adapters are pure transformations: JSONL/JSON records in, plain dicts
out, strict validation with the offending record named in every error.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

Records = str | Path | Iterable[dict]

# RAGTruth label_type -> agreement-fixture gold verdict. Conflict labels mean
# the source states otherwise; baseless labels mean the source is silent.
RAGTRUTH_GOLD = {
    "Evident Conflict": "contradicted",
    "Subtle Conflict": "contradicted",
    "Evident Baseless Info": "unsupported",
    "Subtle Baseless Info": "unsupported",
}

_PASSAGE_SPLIT_RE = re.compile(r"passage\s+\d+\s*:", re.IGNORECASE)


def _load_records(records: Records, what: str) -> list[dict]:
    """Read records from a JSONL path or pass an iterable of dicts through.

    Args:
        records: Path to a JSONL file, or an iterable of decoded dicts.
        what: Human-readable record kind for error messages.

    Returns:
        The decoded records, in input order.

    Raises:
        ValueError: On invalid JSON or a non-object record; the message names
            the record kind and line number.
    """
    if isinstance(records, (str, Path)):
        decoded: list[dict] = []
        with open(records, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    decoded.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{what} line {lineno}: invalid JSON: {e}") from e
    else:
        decoded = list(records)
    for i, rec in enumerate(decoded, start=1):
        if not isinstance(rec, dict):
            raise ValueError(f"{what} record {i}: must be a JSON object")
    return decoded


def _passages(source_info: object, source_id: str) -> list[str]:
    """Extract source passages from a RAGTruth QA ``source_info`` value.

    RAGTruth ships passages either as a list of strings or as one string with
    "passage N:" headers; both forms are accepted.

    Args:
        source_info: The record's ``source_info`` value.
        source_id: Record id for error messages.

    Returns:
        Non-empty passage texts, in source order.

    Raises:
        ValueError: If no passages can be extracted.
    """
    raw: object
    if isinstance(source_info, dict):
        raw = source_info.get("passages")
    else:
        raw = source_info
    passages: list[str] = []
    if isinstance(raw, list):
        passages = [p.strip() for p in raw if isinstance(p, str) and p.strip()]
    elif isinstance(raw, str):
        parts = _PASSAGE_SPLIT_RE.split(raw)
        passages = [p.strip() for p in parts if p.strip()]
    if not passages:
        raise ValueError(f"ragtruth source {source_id!r}: no passages found")
    return passages


def _ragtruth_join(
    source_records: Records, response_records: Records, task_types: tuple[str, ...]
) -> list[tuple[dict, dict, list[str]]]:
    """Join responses to their sources and extract passages.

    Args:
        source_records: RAGTruth ``source_info`` records (path or iterable).
        response_records: RAGTruth ``response`` records (path or iterable).
        task_types: Task types to keep (QA records have a question to pose).

    Returns:
        One (source, response, passages) triple per kept response.

    Raises:
        ValueError: If a response references an unknown source id, or a kept
            source has no question or passages.
    """
    sources = {
        str(rec.get("source_id")): rec
        for rec in _load_records(source_records, "ragtruth source")
    }
    joined: list[tuple[dict, dict, list[str]]] = []
    for rec in _load_records(response_records, "ragtruth response"):
        source_id = str(rec.get("source_id"))
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"ragtruth response references unknown source {source_id!r}")
        if source.get("task_type") not in task_types:
            continue
        info = source.get("source_info")
        question = info.get("question") if isinstance(info, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"ragtruth source {source_id!r}: no question")
        joined.append((source, rec, _passages(info, source_id)))
    return joined


def ragtruth_dataset_rows(
    source_records: Records,
    response_records: Records,
    *,
    task_types: tuple[str, ...] = ("QA",),
) -> list[dict]:
    """Convert RAGTruth QA records into runner dataset rows.

    Args:
        source_records: RAGTruth ``source_info`` records (path or iterable).
        response_records: RAGTruth ``response`` records (path or iterable).
        task_types: Task types to keep; defaults to QA, the grounded-answer
            shape this pipeline evaluates.

    Returns:
        Rows for ``runner.run_dataset``. Each passage becomes a doc with id
        ``p<N>``, so positional markers resolve if a response happens to cite;
        RAGTruth answers are typically uncited, which the citation-recall
        metric reports rather than hides.

    Raises:
        ValueError: On malformed records (the message names the record).
    """
    rows: list[dict] = []
    for source, response, passages in _ragtruth_join(
        source_records, response_records, task_types
    ):
        answer = response.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"ragtruth response {response.get('id')!r}: empty response text"
            )
        rows.append(
            {
                "id": f"{source['source_id']}:{response.get('id', response.get('model', 'r'))}",
                "question": source["source_info"]["question"],
                "answer": answer,
                "docs": [
                    {
                        "doc_id": f"p{i + 1}",
                        "title": f"passage {i + 1}",
                        "text": text,
                    }
                    for i, text in enumerate(passages)
                ],
            }
        )
    return rows


def ragtruth_fixture_rows(
    source_records: Records,
    response_records: Records,
    *,
    task_types: tuple[str, ...] = ("QA",),
) -> list[dict]:
    """Convert RAGTruth span labels into agreement fixture rows.

    Each human-labelled hallucination span becomes one fixture: the span text
    is the claim, the source passages are the spans, and the gold verdict is
    mapped through ``RAGTRUTH_GOLD``. Labelled spans are response fragments,
    not decontextualized claims — fixtures inherit that limitation, which the
    per-tag agreement breakdown makes visible (tag ``ragtruth``).

    Args:
        source_records: RAGTruth ``source_info`` records (path or iterable).
        response_records: RAGTruth ``response`` records (path or iterable).
        task_types: Task types to keep.

    Returns:
        Rows loadable by ``agreement.load_fixtures``.

    Raises:
        ValueError: On malformed records or a label type outside
            ``RAGTRUTH_GOLD`` (fail loudly rather than mislabel gold data).
    """
    fixtures: list[dict] = []
    for _, response, passages in _ragtruth_join(
        source_records, response_records, task_types
    ):
        labels = response.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"ragtruth response {response.get('id')!r}: bad labels")
        for k, label in enumerate(labels):
            label_type = label.get("label_type") if isinstance(label, dict) else None
            text = label.get("text") if isinstance(label, dict) else None
            if label_type not in RAGTRUTH_GOLD:
                raise ValueError(
                    f"ragtruth response {response.get('id')!r}: "
                    f"unknown label_type {label_type!r}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"ragtruth response {response.get('id')!r}: label {k} has no text"
                )
            fixtures.append(
                {
                    "id": f"rt-{response.get('source_id')}-{response.get('id', 'r')}-{k}",
                    "claim": text.strip(),
                    "spans": passages,
                    "gold": RAGTRUTH_GOLD[label_type],
                    "tags": ["ragtruth", label_type.lower().replace(" ", "-")],
                }
            )
    return fixtures


def alce_dataset_rows(records: Records) -> list[dict]:
    """Convert ALCE result records into runner dataset rows.

    ALCE result files carry, per item, the question, the retrieved docs and
    the model output with ``[1]``-style citation markers — the positional
    convention ``pipeline.evaluate`` resolves natively, so citation recall
    and precision are measured exactly as ALCE defines them.

    Args:
        records: ALCE result records (JSONL path or iterable of dicts), each
            with "question", "output", and "docs" ([{"title", "text"}, ...]).

    Returns:
        Rows for ``runner.run_dataset``; doc ids are 1-based positions as
        strings, matching the markers in the output.

    Raises:
        ValueError: On a record missing question, output, or docs.
    """
    rows: list[dict] = []
    for i, rec in enumerate(_load_records(records, "alce record"), start=1):
        question = rec.get("question")
        output = rec.get("output")
        docs = rec.get("docs")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"alce record {i}: missing question")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"alce record {i}: missing output")
        if not isinstance(docs, list) or not docs:
            raise ValueError(f"alce record {i}: missing docs")
        row_docs = []
        for j, doc in enumerate(docs):
            if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
                raise ValueError(f"alce record {i}: docs[{j}] has no text")
            row_docs.append(
                {
                    "doc_id": str(j + 1),
                    "title": str(doc.get("title", f"doc {j + 1}")),
                    "text": doc["text"],
                }
            )
        rows.append(
            {
                "id": str(rec.get("id", f"alce-{i}")),
                "question": question,
                "answer": output,
                "docs": row_docs,
            }
        )
    return rows
