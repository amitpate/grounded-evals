# grounded-evals

**An evaluation harness for retrieval-grounded generation. A citation is treated as a machine-checkable contract between an answer and its sources — this repo checks the contract.**

Generative systems earn trust when their output is accountable to something real. For a text-to-3D system that means physics and navigability; for a retrieval-grounded answer it means *citations that actually support the claims they are attached to*. Most RAG pipelines measure retrieval quality and answer quality separately, and never verify the link between them. That link is where grounded systems quietly fail: the answer is fluent, the sources are relevant, and the specific sentence a user relies on is supported by none of them.

`grounded-evals` evaluates that link, claim by claim.

## What it measures

Given `(question, generated answer with citations, source corpus)`:

| Stage | What happens |
|---|---|
| **1. Claim extraction** | The answer is decomposed into atomic, checkable claims. Rhetoric, hedges and connective tissue are discarded; factual assertions survive. |
| **2. Citation alignment** | Each claim's citations are resolved to concrete source spans — not "document 3," but the sentences the citation actually points at. Lexical overlap proposes candidate spans; the judge confirms. |
| **3. Faithfulness verdict** | Each claim gets a verdict against its aligned spans: `SUPPORTED`, `PARTIAL` (the span supports a weaker version of the claim), or `UNSUPPORTED`. Partial support is scored separately because it is the most dangerous failure mode — it reads as cited and is almost right. |
| **4. Review routing** | Verdicts carry confidence. Low-confidence verdicts are not silently accepted or rejected — they are routed to a human review queue. The design assumption throughout: automated signals have a boundary, and the system should know where it is. |

## Metrics

- **Citation precision** — of the claims that carry citations, the fraction whose citations support them.
- **Citation recall (coverage)** — the fraction of factual claims that carry at least one citation.
- **Unsupported-claim rate** — claims presented as grounded that no source supports. The headline number; this is what erodes user trust.
- **Partial-support rate** — tracked separately from unsupported, for the reason above.
- **Abstention quality** — whether the system declined to answer when the corpus genuinely does not contain the answer (measured on adversarial questions).
- **Review-queue load** — the fraction of verdicts falling below the confidence threshold. A grounding pipeline that routes 40% of its verdicts to humans does not scale; one that routes 0% is not being honest about its judge.

## Design notes

- **Judges are pluggable.** `judge/llm.py` speaks to any OpenAI- or Anthropic-compatible endpoint; `judge/mock.py` is deterministic and runs the full pipeline in CI with no keys and no network. Verdicts are structured JSON, validated against a schema, retried on mismatch.
- **The judge is also under test.** A small labelled fixture set (claims with known verdicts) measures judge agreement, because an unvalidated LLM judge is just a second opinion with better formatting.
- **Everything is a dataclass.** `Claim`, `SourceSpan`, `Alignment`, `Verdict`, `EvalReport` — the pipeline is inspectable at every stage, and every verdict is traceable back to the exact source text it was judged against.

## Run it

```bash
pip install -e .
python examples/run_eval.py                 # mock judge, bundled corpus — no keys needed
GROUNDED_EVALS_JUDGE=llm \
GROUNDED_EVALS_MODEL=<model> \
GROUNDED_EVALS_API_KEY=<key> \
python examples/run_eval.py                 # real judge
```

Output is an `EvalReport`: per-claim verdicts with aligned spans, aggregate metrics, and the review queue.

## Scope, honestly

This is an evaluation harness, not a serving system. It does not do retrieval, chunking or index management — it assumes you have a grounded answer and asks whether the grounding is real. It is the citation-domain sibling of the evaluation layer I run in production for generative 3D (geometry, collision and navigability checks, model-judged prompt adherence, human review at the automation boundary): a different medium, the same contract.

## Author

Amit Pate — [amitpate.github.io](https://amitpate.github.io)

MIT License

