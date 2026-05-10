"""
TranslationBackend protocol and shared data structures.

All backends implement this interface so the streaming policy
can work with any model without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable


# ── Shared data structures ────────────────────────────────────────────────────

class Direction(Enum):
    GLOSS_TO_TEXT = "gloss_to_text"
    TEXT_TO_GLOSS = "text_to_gloss"


@dataclass
class TranslationResult:
    translation: str
    readiness_score: float = 0.0    # [0, 1] — how ready the model was
    raw_response: str = ""          # unprocessed model output (for debugging)


# ── Backend protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class TranslationBackend(Protocol):
    """
    Minimal interface every translation backend must satisfy.

    Implementations: T5Backend, OllamaBackend, GroqBackend, GeminiBackend.
    """

    @property
    def name(self) -> str:
        """Short identifier used in result tables (e.g. 'T5', 'GPT-OSS')."""
        ...

    @property
    def supports_native_readiness(self) -> bool:
        """
        True if this backend computes the readiness score natively
        (i.e. via encoder hidden states + LRE head — T5 only).
        False means the AFG will use a prompt-based fallback.
        """
        ...

    async def translate(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        """
        Translate a gloss sequence (or English text for text_to_gloss) to the target.

        Args:
            glosses:   Space-separated gloss tokens (or English sentence).
            context:   Preceding translated sentences joined by ' ||| ' for
                       discourse context (may be ignored by API backends).
            direction: Translation direction.
        """
        ...

    async def get_readiness_score(self, glosses: str) -> float:
        """
        Estimate translation readiness for accumulated glosses.

        Returns a score in [0, 1]:
            0 = clearly incomplete
            1 = complete and ready to translate

        T5Backend uses the learned LRE head.
        API backends use a prompt-based confidence query.
        """
        ...

    async def wait_or_translate(self, glosses: str) -> str:
        """
        TransLLaMa-style decision: translate now or wait for more input.

        Returns either:
          - The English translation (→ WRITE decision)
          - "<WAIT>" (→ READ decision)

        T5Backend: delegates to translate() and checks for the trained <WAIT> token.
        API backends: sends a prompt asking the model to output <WAIT> or translate.
        """
        ...

    async def close(self) -> None:
        """Release any resources (HTTP sessions, model memory, etc.)."""
        ...


# ── Shared text cleaning ──────────────────────────────────────────────────────

def clean_model_output(text: str) -> str:
    """
    Strip common LLM artifacts from translation output:
      - Leading labels: "Translation:", "English:", "Answer:", etc.
      - Surrounding quotes
      - Extra whitespace
    """
    if not text:
        return ""
    text = text.strip()

    # Remove common prefixes (case-insensitive)
    prefixes = [
        "translation:", "english:", "answer:", "here is",
        "here's", "output:", "result:", "asl gloss:",
    ]
    lower = text.lower()
    for p in prefixes:
        if lower.startswith(p):
            text = text[len(p):].strip()
            lower = text.lower()

    # Remove surrounding quotes
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1]

    return text.strip()
