"""
Adaptive Fusion Gate (AFG).

Combines the Temporal Pause Detector score and the Linguistic Readiness
Estimator score to produce a READ or WRITE decision.

This is the decision-making core of TLAS.
"""

from dataclasses import dataclass
from enum import Enum

from src.config import cfg


class PolicyDecision(Enum):
    READ  = "read"
    WRITE = "write"


@dataclass
class AFGDecision:
    decision: PolicyDecision
    pause_score: float
    readiness_score: float
    combined_score: float
    reason: str


class AdaptiveFusionGate:
    """
    Fuses temporal and linguistic signals to make the READ/WRITE decision.

    Three trigger conditions for WRITE (evaluated in order):
      1. Safety valve: buffer has accumulated max_lag glosses.
      2. Joint threshold: weighted sum of pause + readiness > combined threshold.
      3. Strong pause override: pause_score is very high AND readiness is above
         a low minimum (the signer has paused, so we trust them even if the
         model is not fully confident).

    READ is returned if none of the above conditions are met.
    """

    def __init__(
        self,
        weight_temporal: float  = None,
        weight_linguistic: float = None,
        threshold: float        = None,
        strong_pause: float     = None,
        min_readiness: float    = None,
        max_lag: int            = None,
    ):
        self.w_t   = weight_temporal   or cfg.tlas.afg_weight_temporal
        self.w_l   = weight_linguistic or cfg.tlas.afg_weight_linguistic
        self.theta = threshold         or cfg.tlas.afg_threshold
        self.strong_pause_thresh = strong_pause  or cfg.tlas.afg_strong_pause_threshold
        self.min_readiness       = min_readiness or cfg.tlas.afg_min_readiness_for_pause
        self.max_lag             = max_lag       or cfg.tlas.afg_max_lag

    def decide(
        self,
        pause_score: float,
        readiness_score: float,
        buffer_len: int,
        is_final: bool = False,
    ) -> AFGDecision:
        """
        Make a READ/WRITE decision.

        Args:
            pause_score:     TPD output in [0, 1].
            readiness_score: LRE output in [0, 1].
            buffer_len:      Number of glosses currently in the buffer.
            is_final:        True if the input stream has ended (force WRITE).
        """
        combined = self.w_t * pause_score + self.w_l * readiness_score

        # Always write on the last gloss
        if is_final:
            return AFGDecision(PolicyDecision.WRITE, pause_score, readiness_score,
                               combined, "final gloss")

        # Safety valve
        if buffer_len >= self.max_lag:
            return AFGDecision(PolicyDecision.WRITE, pause_score, readiness_score,
                               combined, f"max_lag={self.max_lag}")

        # Joint threshold
        if combined >= self.theta:
            return AFGDecision(PolicyDecision.WRITE, pause_score, readiness_score,
                               combined,
                               f"combined={combined:.3f} ≥ θ={self.theta}")

        # Strong pause override
        if (pause_score >= self.strong_pause_thresh
                and readiness_score >= self.min_readiness):
            return AFGDecision(PolicyDecision.WRITE, pause_score, readiness_score,
                               combined,
                               f"strong_pause={pause_score:.3f}, readiness={readiness_score:.3f}")

        return AFGDecision(PolicyDecision.READ, pause_score, readiness_score,
                           combined,
                           f"combined={combined:.3f} < θ={self.theta}")
