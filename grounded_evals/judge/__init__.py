from .base import Judge
from .mock import MockJudge

__all__ = ["Judge", "MockJudge", "LLMJudge"]


def __getattr__(name):
    if name == "LLMJudge":  # lazy: importable without env configured
        from .llm import LLMJudge
        return LLMJudge
    raise AttributeError(name)

