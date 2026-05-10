"""
Temporal Pause Detector (TPD).

Monitors inter-gloss arrival times and computes a pause score in [0, 1].
The score rises when the current gap significantly exceeds the running average
— indicating a natural sentence boundary in the signing stream.

This is the temporal half of the TLAS novel contribution.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from src.config import cfg


@dataclass
class TPDState:
    """Per-stream state; create one per active speaker."""
    ema_delta: float = 0.0       # exponential moving average of Δt
    last_timestamp: Optional[float] = None
    gloss_count: int = 0


class TemporalPauseDetector:
    """
    Computes a pause score from inter-gloss timing.

    Algorithm:
        1. Maintain an EMA of recent inter-gloss intervals Δt.
        2. When a new gloss arrives at time t:
               Δt = t - t_prev
               ema = α·Δt + (1-α)·ema
               ratio = Δt / ema
        3. pause_score = clamp((ratio - 1) / (multiplier - 1), 0, 1)

        ratio = 1   → normal pace   → pause_score = 0
        ratio ≥ multiplier → long pause → pause_score = 1

    The EMA naturally adapts to each signer's pace, so the detector fires
    on *relative* pauses rather than absolute ones.

    The EMA is initialized with a prior (default 450ms) derived from the
    average inter-gloss gap in training data, avoiding cold-start
    fluctuations in the first few observations.
    """

    def __init__(
        self,
        alpha: float = None,
        pause_multiplier: float = None,
        min_buffer: int = None,
    ):
        self.alpha            = alpha            or cfg.tlas.tpd_alpha
        self.pause_multiplier = pause_multiplier or cfg.tlas.tpd_pause_multiplier
        self.min_buffer       = min_buffer       or cfg.tlas.tpd_min_buffer
        self._ema_prior       = cfg.tlas.tpd_ema_prior_ms / 1000.0   # seconds
        self._state = TPDState(ema_delta=self._ema_prior)

    def update(self, timestamp: float) -> float:
        """
        Register a new gloss arrival and return the pause score for the
        *preceding* inter-gloss gap.

        Returns:
            pause_score in [0, 1]
        """
        self._state.gloss_count += 1
        return self._step(self._state, timestamp)

    def compute_from_timestamps(self, timestamps: List[float]) -> float:
        """
        Compute pause score for a buffer of glosses given their timestamps.

        Returns the pause score based on the *most recent* inter-gloss gap
        in the buffer (i.e., the gap just before the last gloss).
        """
        if len(timestamps) < 2:
            return 0.0
        tmp = TPDState(ema_delta=self._ema_prior)
        score = 0.0
        for ts in timestamps:
            score = self._step(tmp, ts)
        return score

    def _step(self, state: TPDState, timestamp: float) -> float:
        """Core computation shared by update() and compute_from_timestamps()."""
        if state.last_timestamp is None:
            state.last_timestamp = timestamp
            return 0.0

        delta = timestamp - state.last_timestamp
        state.last_timestamp = timestamp

        state.ema_delta = self.alpha * delta + (1.0 - self.alpha) * state.ema_delta
        ratio = delta / (state.ema_delta + 1e-9)
        raw = (ratio - 1.0) / (self.pause_multiplier - 1.0)
        return max(0.0, min(1.0, raw))

    def reset(self):
        """Reset state between sentences/speakers."""
        self._state = TPDState(ema_delta=self._ema_prior)

    @property
    def current_ema(self) -> Optional[float]:
        return self._state.ema_delta
