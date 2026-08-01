"""Run the full pipeline on a bundled corpus with the mock judge (no keys),
or the LLM judge when GROUNDED_EVALS_JUDGE=llm is set.

The example answer contains, on purpose: a supported claim, a contradicted
claim (right fact, wrong number — the sneakiest failure), an unsupported
cited claim, an uncited claim, and an opinion — one of each failure mode the
metrics separate. Citations use numbered markers ("[1]", "[2]") the way real
grounded generators emit them; the pipeline resolves them against the source
list.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grounded_evals import SourceDoc, evaluate
from grounded_evals.judge import MockJudge

CORPUS = [
    SourceDoc(
        "wembley",
        "Wembley Stadium factsheet",
        "Wembley Stadium in London has a seating capacity of 90,000, making it "
        "the largest stadium in the United Kingdom. The stadium reopened in "
        "2007 after a complete rebuild. The arch above the stadium spans 315 "
        "metres and is visible across much of London.",
    ),
    SourceDoc(
        "etihad",
        "Etihad Stadium factsheet",
        "The Etihad Stadium is the home ground of Manchester City Football "
        "Club. Its capacity is approximately 53,400 for football matches. The "
        "stadium hosted athletics during the 2002 Commonwealth Games before "
        "its conversion for football.",
    ),
]

QUESTION = "Compare Wembley Stadium and the Etihad Stadium."

ANSWER = (
    "Wembley Stadium has a seating capacity of 90,000 and is the largest "
    "stadium in the United Kingdom. [1] "
    "The stadium reopened in 2010 after a complete rebuild. [1] "
    "The Etihad Stadium hosts around 80,000 fans for football matches. [2] "
    "The Etihad's naming rights are the most valuable in the Premier League. "
    "I think Wembley is the more iconic venue of the two."
)


def main() -> None:
    if os.environ.get("GROUNDED_EVALS_JUDGE") == "llm":
        from grounded_evals.judge import LLMJudge

        judge = LLMJudge.from_env()
    else:
        judge = MockJudge()

    report = evaluate(QUESTION, ANSWER, CORPUS, judge)

    print(json.dumps(report.summary(), indent=2))
    print()
    for r in report.results:
        cites = ",".join(c.doc_id for c in r.claim.citations) or "-"
        flag = " -> REVIEW" if r.needs_review else ""
        print(f"[{r.verdict.value:>13}] ({cites}) {r.claim.text[:70]}{flag}")
        print(f"                 {r.rationale[:100]}")


if __name__ == "__main__":
    main()
