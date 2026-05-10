"""
TLAS — Temporal-Linguistic Adaptive Streaming policy.

Orchestrates the three components:
  TPD  (Temporal Pause Detector)
  LRE  (Linguistic Readiness Estimator)
  AFG  (Adaptive Fusion Gate)

Supports three ablation modes:
  "full"            → TPD + LRE + AFG  (the full TLAS)
  "temporal_only"   → TPD + AFG (w_l = 0, w_t = 1)
  "linguistic_only" → LRE + AFG (w_t = 0, w_l = 1)  ← equivalent to improved LSG
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from src.policy.afg import AdaptiveFusionGate, AFGDecision, PolicyDecision
from src.policy.lre import LinguisticReadinessEstimator
from src.policy.tpd import TemporalPauseDetector
from src.config import cfg

if TYPE_CHECKING:
    from src.backends.base import TranslationBackend

logger = logging.getLogger(__name__)


class TLASMode(Enum):
    FULL             = "full"
    TEMPORAL_ONLY    = "temporal_only"
    LINGUISTIC_ONLY  = "linguistic_only"


@dataclass
class TLASStep:
    """Record of a single TLAS decision step (useful for analysis / paper figures)."""
    buffer:           List[str]
    timestamps:       List[float]
    pause_score:      float
    readiness_score:  float
    combined_score:   float
    decision:         PolicyDecision
    reason:           str


class TLASPolicy:
    """
    The TLAS streaming policy.

    Usage:
        policy = TLASPolicy(backend, mode=TLASMode.FULL)
        policy.reset()

        for gloss, ts in stream:
            decision = await policy.step(gloss, ts)
            if decision.decision == PolicyDecision.WRITE:
                translation = await backend.translate(" ".join(policy.buffer))
                policy.flush()
    """

    def __init__(
        self,
        backend: "TranslationBackend",
        mode: TLASMode = TLASMode.FULL,
    ):
        self.backend = backend
        self.mode    = mode

        self.tpd = TemporalPauseDetector()
        self.lre = LinguisticReadinessEstimator(backend)

        # Build AFG with mode-specific weight overrides
        if mode == TLASMode.TEMPORAL_ONLY:
            self.afg = AdaptiveFusionGate(weight_temporal=1.0, weight_linguistic=0.0)
        elif mode == TLASMode.LINGUISTIC_ONLY:
            self.afg = AdaptiveFusionGate(weight_temporal=0.0, weight_linguistic=1.0)
        else:
            self.afg = AdaptiveFusionGate()

        self.buffer: List[str]    = []
        self.timestamps: List[float] = []
        self.history: List[TLASStep] = []

    async def step(
        self,
        gloss: str,
        timestamp: float,
        is_final: bool = False,
    ) -> AFGDecision:
        """
        Process one incoming gloss and return the AFG decision.

        Args:
            gloss:      The new gloss token (upper-case).
            timestamp:  Wall-clock time (seconds since stream start).
            is_final:   True if this is the last gloss in the stream.
        """
        self.buffer.append(gloss)
        self.timestamps.append(timestamp)

        # TPD: score the gap just before this gloss
        # In LINGUISTIC_ONLY mode, discard the temporal signal so the
        # ablation measures pure linguistic readiness (symmetric with
        # TEMPORAL_ONLY which discards LRE).
        if self.mode == TLASMode.LINGUISTIC_ONLY:
            self.tpd.update(timestamp)   # keep EMA updated (unused)
            pause_score = 0.0
        else:
            pause_score = self.tpd.update(timestamp)

        # LRE: estimate readiness from accumulated glosses
        if self.mode == TLASMode.TEMPORAL_ONLY:
            readiness_score = 0.0
        else:
            glosses_str = " ".join(self.buffer)
            readiness_score = await self.lre.score(glosses_str)

        # AFG: combine and decide
        decision = self.afg.decide(
            pause_score=pause_score,
            readiness_score=readiness_score,
            buffer_len=len(self.buffer),
            is_final=is_final,
        )

        self.history.append(TLASStep(
            buffer=list(self.buffer),
            timestamps=list(self.timestamps),
            pause_score=pause_score,
            readiness_score=readiness_score,
            combined_score=decision.combined_score,
            decision=decision.decision,
            reason=decision.reason,
        ))

        return decision

    def flush(self):
        """Clear the buffer after a WRITE decision."""
        self.buffer.clear()
        self.timestamps.clear()

    def reset(self):
        """Full reset between sentences or at stream start."""
        self.flush()
        self.tpd.reset()
        self.lre.reset()
        self.history.clear()

    @property
    def name(self) -> str:
        labels = {
            TLASMode.FULL:            "TLAS",
            TLASMode.TEMPORAL_ONLY:   "TLAS-temporal",
            TLASMode.LINGUISTIC_ONLY: "TLAS-linguistic",
        }
        return labels[self.mode]
