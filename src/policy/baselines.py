"""
Streaming policies for comparison with TLAS.

External baselines (published methods):
  BatchPolicy       — oracle: translate complete sentence at once (upper bound)
  WaitKPolicy       — fixed k-token delay  (Ma et al., 2019)
  TransLLaMaPolicy  — learned WAIT/translate decision  (Agostinelli et al., 2023)

Ablations (this work — not published external methods):
  LSGPolicy         — KL divergence + confidence threshold, our prior approach
                      before training the LRE head. Included to show the
                      improvement from the learned LRE over raw encoder heuristics.

All share the same async interface as TLASPolicy for drop-in evaluation.
"""

import asyncio
import logging
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.policy.afg import PolicyDecision, AFGDecision
from src.policy.lre import LinguisticReadinessEstimator
from src.config import cfg

if TYPE_CHECKING:
    from src.backends.base import TranslationBackend
    from src.backends.t5_backend import T5Backend

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _afg(decision: PolicyDecision, reason: str = "") -> AFGDecision:
    """Create an AFGDecision with dummy scores (baselines don't use them)."""
    return AFGDecision(decision, 0.0, 0.0, 0.0, reason)


# ── Batch (oracle) ────────────────────────────────────────────────────────────

class BatchPolicy:
    """Always returns WRITE only on the final gloss.  Upper-bound baseline."""

    name = "Batch"

    def __init__(self, *args, **kwargs):
        self.buffer: List[str] = []
        self.timestamps: List[float] = []

    async def step(self, gloss: str, timestamp: float, is_final: bool = False) -> AFGDecision:
        self.buffer.append(gloss)
        self.timestamps.append(timestamp)
        if is_final:
            return _afg(PolicyDecision.WRITE, "final")
        return _afg(PolicyDecision.READ)

    def flush(self):
        self.buffer.clear()
        self.timestamps.clear()

    def reset(self):
        self.flush()


# ── Wait-k ────────────────────────────────────────────────────────────────────

class WaitKPolicy:
    """
    Fixed k-token delay.  After accumulating k glosses, translates every k glosses
    (non-overlapping windows). On the final gloss, always flushes any remainder.
    """

    def __init__(self, k: int = None, *args, **kwargs):
        self.k = k or cfg.baselines.wait_k
        self.name = f"Wait-k (k={self.k})"
        self.buffer: List[str] = []
        self.timestamps: List[float] = []

    async def step(self, gloss: str, timestamp: float, is_final: bool = False) -> AFGDecision:
        self.buffer.append(gloss)
        self.timestamps.append(timestamp)
        if len(self.buffer) >= self.k or is_final:
            return _afg(PolicyDecision.WRITE, f"buf={len(self.buffer)} ≥ k={self.k}")
        return _afg(PolicyDecision.READ)

    def flush(self):
        self.buffer.clear()
        self.timestamps.clear()

    def reset(self):
        self.flush()


# ── TransLLaMa ────────────────────────────────────────────────────────────────

class TransLLaMaPolicy:
    """
    Asks the model to output <WAIT> or a translation at each step.

    For T5: checks if the fine-tuned model emits the trained <WAIT> token.
    For API backends: sends a WAIT-or-translate prompt (one API call per step),
    faithfully replicating the TransLLaMa mechanism without using our LRE.
    """

    name = "TransLLaMa"

    def __init__(self, backend: "TranslationBackend", min_buffer: int = 2):
        self.backend    = backend
        self.min_buffer = min_buffer
        self.buffer: List[str] = []
        self.timestamps: List[float] = []

    async def step(self, gloss: str, timestamp: float, is_final: bool = False) -> AFGDecision:
        self.buffer.append(gloss)
        self.timestamps.append(timestamp)

        if is_final:
            return _afg(PolicyDecision.WRITE, "final")

        if len(self.buffer) < self.min_buffer:
            return _afg(PolicyDecision.READ, f"buf < {self.min_buffer}")

        try:
            raw = await self.backend.wait_or_translate(" ".join(self.buffer))
            if "<WAIT>" in raw or not raw.strip():
                return _afg(PolicyDecision.READ, "<WAIT>")
            return _afg(PolicyDecision.WRITE, "translate")
        except Exception:
            return _afg(PolicyDecision.READ)

    def flush(self):
        self.buffer.clear()
        self.timestamps.clear()

    def reset(self):
        self.flush()


# ── LSG — Ablation (this work, prior version) ─────────────────────────────────
# NOT an external published baseline. LSG is our own earlier approach that uses
# raw T5 encoder probability heuristics (KL divergence + confidence) instead of
# the trained LRE head. Included as an ablation to quantify the benefit of the
# learned LRE over naive encoder statistics.

class LSGPolicy:
    """
    Local Scoring Gate using KL divergence + confidence from T5 encoder.

    For API backends: falls back to prompt-based confidence only.
    For T5 backend: uses model's next-token probability distribution.
    """

    name = "LSG"

    def __init__(
        self,
        backend: "TranslationBackend",
        kl_threshold: float = None,
        conf_threshold: float = None,
        max_lag: int = None,
        baseline_k: int = None,
    ):
        self.backend      = backend
        self.kl_threshold = kl_threshold  or cfg.baselines.lsg_kl_threshold
        self.conf_thresh  = conf_threshold or cfg.baselines.lsg_confidence_threshold
        self.max_lag      = max_lag        or cfg.baselines.lsg_max_lag
        self.baseline_k   = baseline_k     or cfg.baselines.lsg_baseline_k

        self.buffer: List[str] = []
        self.timestamps: List[float] = []
        self._baseline_probs: Optional[torch.Tensor] = None
        self.lre = LinguisticReadinessEstimator(backend)

    def _get_t5_probs(self, glosses: List[str]) -> Optional[torch.Tensor]:
        """Return next-token probs from T5 encoder (sync, call in executor)."""
        try:
            from src.backends.t5_backend import T5Backend
            if not isinstance(self.backend, T5Backend):
                return None
            b = self.backend
            source = b._build_source(" ".join(glosses))
            inputs = b.tokenizer(source, return_tensors="pt", truncation=True).to(b.device)
            decoder_input = torch.tensor(
                [[b.model.config.decoder_start_token_id]], device=b.device
            )
            with torch.no_grad():
                out = b.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input,
                )
            return F.softmax(out.logits[0, -1, :], dim=-1)
        except Exception:
            return None

    @staticmethod
    def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
        eps = 1e-10
        return float(torch.sum((p + eps) * torch.log((p + eps) / (q + eps))).item())

    async def step(self, gloss: str, timestamp: float, is_final: bool = False) -> AFGDecision:
        self.buffer.append(gloss)
        self.timestamps.append(timestamp)

        if is_final:
            return _afg(PolicyDecision.WRITE, "final")

        if len(self.buffer) < self.baseline_k:
            return _afg(PolicyDecision.READ, f"buf < {self.baseline_k}")

        if len(self.buffer) >= self.max_lag:
            return _afg(PolicyDecision.WRITE, f"max_lag={self.max_lag}")

        # T5 path: KL divergence + confidence
        loop = asyncio.get_event_loop()
        p_curr = await loop.run_in_executor(None, self._get_t5_probs, self.buffer)

        if p_curr is not None:
            if self._baseline_probs is None:
                baseline = self.buffer[:self.baseline_k]
                self._baseline_probs = await loop.run_in_executor(
                    None, self._get_t5_probs, baseline
                )

            if self._baseline_probs is not None:
                kl  = self._kl(p_curr, self._baseline_probs)
                conf = float(p_curr.max().item())
                if kl > self.kl_threshold:
                    return _afg(PolicyDecision.WRITE, f"KL={kl:.3f}")
                if conf > self.conf_thresh:
                    return _afg(PolicyDecision.WRITE, f"conf={conf:.3f}")
        else:
            # API fallback: use LRE (standalone T5 encoder, avoids per-gloss API calls)
            score = await self.lre.score(" ".join(self.buffer))
            if score > self.conf_thresh:
                return _afg(PolicyDecision.WRITE, f"lre_conf={score:.3f}")

        return _afg(PolicyDecision.READ)

    def flush(self):
        self.buffer.clear()
        self.timestamps.clear()
        self._baseline_probs = None

    def reset(self):
        self.flush()


# ── Factory ───────────────────────────────────────────────────────────────────

def get_all_baselines(backend: "TranslationBackend") -> list:
    """Return all baseline policy instances for a given backend."""
    from src.policy.tlas import TLASPolicy, TLASMode
    return [
        BatchPolicy(),
        WaitKPolicy(k=cfg.baselines.wait_k),
        TransLLaMaPolicy(backend),
        LSGPolicy(backend),
        TLASPolicy(backend, mode=TLASMode.FULL),
        TLASPolicy(backend, mode=TLASMode.TEMPORAL_ONLY),
        TLASPolicy(backend, mode=TLASMode.LINGUISTIC_ONLY),
    ]
