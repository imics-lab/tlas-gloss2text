"""
Linguistic Readiness Estimator (LRE).

Wraps the LREHead neural network (defined in t5_backend.py) for use within
the streaming policy.

For T5 backends: delegates to backend.get_readiness_score() which uses the
encoder hidden states + trained LRE head directly.

For non-T5 backends (Gemini, Ollama, etc.): loads the T5 encoder + LRE head
locally as a standalone scorer. This avoids expensive per-gloss API calls
while keeping the same trained neural readiness estimation.

This is the linguistic half of the TLAS novel contribution.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.backends.base import TranslationBackend

logger = logging.getLogger(__name__)


# ── Standalone LRE scorer (singleton, shared across all LRE instances) ────────

_standalone_scorer = None
_standalone_scorer_loaded = False


def _load_standalone_scorer():
    """
    Load the T5 encoder + LRE head as a standalone readiness scorer.

    Called once on first use; the loaded model is cached in module globals
    so that multiple TLAS policy instances (TLAS-full, TLAS-linguistic)
    share the same encoder without duplicating GPU memory.
    """
    global _standalone_scorer, _standalone_scorer_loaded

    if _standalone_scorer_loaded:
        return _standalone_scorer

    _standalone_scorer_loaded = True

    try:
        import torch
        from transformers import T5TokenizerFast, T5ForConditionalGeneration
        from src.backends.t5_backend import LREHead
        from src.config import cfg

        # Find checkpoint directory with LRE head
        checkpoint_dir = Path(cfg.t5.checkpoint_dir)
        lre_path = checkpoint_dir / "lre_head.pt"

        if not lre_path.exists():
            # Search common locations
            for candidate in [
                "models/t5_base_discourse/final",
                "models/t5_model",
            ]:
                p = Path(candidate) / "lre_head.pt"
                if p.exists():
                    lre_path = p
                    checkpoint_dir = Path(candidate)
                    break

        if not lre_path.exists():
            logger.warning(
                "No LRE head found for standalone readiness scoring. "
                "LRE will return 0.5 for all inputs."
            )
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Loading standalone T5 encoder + LRE head from {checkpoint_dir} "
            f"on {device} for non-T5 backend readiness scoring..."
        )

        tokenizer = T5TokenizerFast.from_pretrained(str(checkpoint_dir))

        # Load full model, extract encoder, discard decoder to save memory
        full_model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_dir))
        encoder = full_model.get_encoder().to(device).eval()

        hidden_dim = full_model.config.d_model
        lre_head = LREHead(hidden_dim=hidden_dim)
        lre_head.load_state_dict(torch.load(lre_path, map_location=device))
        lre_head.to(device).eval()

        # Free decoder weights
        del full_model

        prefix = cfg.t5.gloss_to_text_prefix  # "translate ASL to English: "

        _standalone_scorer = {
            "tokenizer": tokenizer,
            "encoder": encoder,
            "lre_head": lre_head,
            "device": device,
            "prefix": prefix,
        }
        logger.info("Standalone LRE scorer ready.")
        return _standalone_scorer

    except Exception as e:
        logger.error(f"Failed to load standalone LRE scorer: {e}")
        return None


# ── LRE class ─────────────────────────────────────────────────────────────────

class LinguisticReadinessEstimator:
    """
    Estimates how ready accumulated glosses are for translation.

    Strategy:
      - If the backend supports native readiness (T5 with LRE head) →
        delegate to backend.get_readiness_score()
      - Otherwise → use the standalone T5 encoder + LRE head loaded locally

    The readiness score is in [0, 1]:
        0 = clearly incomplete (need more glosses)
        1 = complete and ready to translate
    """

    def __init__(self, backend: "TranslationBackend"):
        self.backend = backend
        self._use_standalone = not backend.supports_native_readiness

        if self._use_standalone:
            # Trigger lazy load of the singleton scorer
            _load_standalone_scorer()

    async def score(self, glosses: str) -> float:
        """Return a readiness score in [0, 1] for the accumulated gloss buffer."""
        if not glosses.strip():
            return 0.0

        if self._use_standalone:
            return self._score_standalone(glosses)

        try:
            return await self.backend.get_readiness_score(glosses)
        except Exception as e:
            logger.warning(f"LRE score failed: {e}. Returning 0.5.")
            return 0.5

    def _score_standalone(self, glosses: str) -> float:
        """Score readiness using the locally loaded T5 encoder + LRE head."""
        import torch

        scorer = _standalone_scorer
        if scorer is None:
            return 0.5

        source = f"{scorer['prefix']}{glosses}"
        inputs = scorer["tokenizer"](
            source,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to(scorer["device"])

        with torch.no_grad():
            encoder_out = scorer["encoder"](
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            score = scorer["lre_head"](
                encoder_out.last_hidden_state,
                inputs["attention_mask"],
            )
        return float(score.item())

    def reset(self):
        pass
