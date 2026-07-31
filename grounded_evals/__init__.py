"""grounded-evals: citation faithfulness evaluation for retrieval-grounded generation."""
from .pipeline import evaluate
from .schema import Citation, Claim, EvalReport, SourceDoc, Verdict

__all__ = ["evaluate", "Citation", "Claim", "EvalReport", "SourceDoc", "Verdict"]
__version__ = "0.1.0"

