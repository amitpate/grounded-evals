"""grounded-evals: citation faithfulness evaluation for retrieval-grounded generation."""
from .pipeline import evaluate, marker_map
from .schema import (
    Alignment,
    Citation,
    Claim,
    ClaimResult,
    EvalReport,
    JudgeVerdict,
    SourceDoc,
    SourceSpan,
    Verdict,
)

__all__ = [
    "evaluate",
    "marker_map",
    "Alignment",
    "Citation",
    "Claim",
    "ClaimResult",
    "EvalReport",
    "JudgeVerdict",
    "SourceDoc",
    "SourceSpan",
    "Verdict",
]
__version__ = "0.2.0"
