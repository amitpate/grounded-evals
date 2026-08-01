from typing import Any

from .base import Judge, JudgeError
from .mock import MockJudge

__all__ = ["Judge", "JudgeError", "MockJudge", "LLMJudge"]


def __getattr__(name: str) -> Any:
    if name == "LLMJudge":  # lazy: importable without transport concerns
        from .llm import LLMJudge

        return LLMJudge
    raise AttributeError(name)
