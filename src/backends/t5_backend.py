"""
T5 translation backend with optional Linguistic Readiness Estimator (LRE) head.

The LRE head is a small MLP that attaches to the T5 encoder and predicts
translation readiness in [0, 1].
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.backends.base import Direction, TranslationBackend, TranslationResult, clean_model_output
from src.config import cfg

logger = logging.getLogger(__name__)


# ── LRE Head ─────────────────────────────────────────────────────────────────

class LREHead(nn.Module):
    """
    Linguistic Readiness Estimator.

    Takes pooled T5 encoder hidden states → scalar readiness score in [0, 1].
    Trained separately after T5 fine-tuning (see training/train_lre.py).
    """

    def __init__(self, hidden_dim: int = 768, intermediate: int = None):
        super().__init__()
        intermediate = intermediate or cfg.tlas.lre_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.ReLU(),
            nn.Dropout(cfg.tlas.lre_dropout),
            nn.Linear(intermediate, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,   # [batch, seq_len, hidden]
        attention_mask: torch.Tensor,           # [batch, seq_len]
    ) -> torch.Tensor:                          # [batch]
        # Mean-pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoder_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.net(pooled).squeeze(-1)


# ── T5 Backend ────────────────────────────────────────────────────────────────

class T5Backend:
    """
    Fine-tuned T5 translation backend.

    Supports both gloss→text and text→gloss via task prefixes.
    When the LRE head is loaded, get_readiness_score() uses it directly;
    otherwise it falls back to a heuristic based on output entropy.
    """

    def __init__(
        self,
        checkpoint_dir: str = None,
        device: str = None,
        max_workers: int = 1,
    ):
        self.checkpoint_dir = Path(checkpoint_dir or cfg.t5.checkpoint_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"T5Backend: loading from {self.checkpoint_dir} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = T5TokenizerFast.from_pretrained(str(self.checkpoint_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(str(self.checkpoint_dir))
        self.model.to(self.device)
        self.model.eval()

        # Register special tokens
        self._add_special_tokens()

        # LRE head (optional — loaded separately)
        self.lre_head: Optional[LREHead] = None
        lre_path = self.checkpoint_dir / "lre_head.pt"
        if lre_path.exists():
            hidden = self.model.config.d_model
            self.lre_head = LREHead(hidden_dim=hidden)
            self.lre_head.load_state_dict(torch.load(lre_path, map_location=self.device))
            self.lre_head.to(self.device)
            self.lre_head.eval()
            logger.info("LRE head loaded.")
        else:
            logger.info("No LRE head found; readiness will use entropy fallback.")

        # Executor for running blocking torch calls from async contexts
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _add_special_tokens(self):
        special = {"additional_special_tokens": [cfg.t5.wait_token]}
        self.tokenizer.add_special_tokens(special)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.wait_token_id = self.tokenizer.convert_tokens_to_ids(cfg.t5.wait_token)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "T5"

    @property
    def supports_native_readiness(self) -> bool:
        return self.lre_head is not None

    # ── Synchronous inference (runs in executor) ───────────────────────────────

    def _build_source(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> str:
        prefix = (
            cfg.t5.gloss_to_text_prefix
            if direction == Direction.GLOSS_TO_TEXT
            else cfg.t5.text_to_gloss_prefix
        )
        if context:
            return f"{prefix}[Context: {context}] {glosses}"
        return f"{prefix}{glosses}"

    @torch.no_grad()
    def _translate_sync(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        source = self._build_source(glosses, context, direction)
        inputs = self.tokenizer(
            source,
            return_tensors="pt",
            max_length=cfg.t5.max_source_length,
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=cfg.t5.max_target_length,
            num_beams=cfg.t5.num_beams,
            no_repeat_ngram_size=cfg.t5.no_repeat_ngram_size,
            early_stopping=cfg.t5.early_stopping,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return TranslationResult(translation=clean_model_output(decoded))

    @torch.no_grad()
    def _readiness_sync(self, glosses: str) -> float:
        source = self._build_source(glosses)
        inputs = self.tokenizer(
            source,
            return_tensors="pt",
            max_length=cfg.t5.max_source_length,
            truncation=True,
        ).to(self.device)

        if self.lre_head is not None:
            # Use trained LRE head on encoder hidden states
            encoder_out = self.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            score = self.lre_head(
                encoder_out.last_hidden_state,
                inputs["attention_mask"],
            )
            return float(score.item())
        else:
            # Entropy-based fallback: low entropy on first decoder token → high readiness
            decoder_input = torch.tensor(
                [[self.model.config.decoder_start_token_id]], device=self.device
            )
            out = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input,
            )
            logits = out.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp(min=1e-10).log()).sum()
            max_entropy = torch.log(torch.tensor(float(logits.shape[0])))
            # Low entropy → high confidence → high readiness
            return float(1.0 - (entropy / max_entropy).clamp(0, 1))

    # ── Async interface ───────────────────────────────────────────────────────

    async def _run_in_executor(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def translate(
        self,
        glosses: str,
        context: str = "",
        direction: Direction = Direction.GLOSS_TO_TEXT,
    ) -> TranslationResult:
        return await self._run_in_executor(self._translate_sync, glosses, context, direction)

    async def get_readiness_score(self, glosses: str) -> float:
        return await self._run_in_executor(self._readiness_sync, glosses)

    async def wait_or_translate(self, glosses: str) -> str:
        """T5: generate and return raw output (may contain <WAIT> token)."""
        result = await self.translate(glosses)
        return result.translation

    async def close(self):
        self._executor.shutdown(wait=False)

    # ── Factory: load from pretrained (before fine-tuning) ───────────────────

    @classmethod
    def from_pretrained(cls, model_name: str = None, **kwargs) -> "T5Backend":
        """
        Create a T5Backend from a HuggingFace model name (not a local checkpoint).
        Used before fine-tuning exists; saves a temporary checkpoint first.
        """
        model_name = model_name or cfg.t5.model_name
        tmp_dir = Path(cfg.t5.checkpoint_dir) / "_pretrained_init"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tok = T5TokenizerFast.from_pretrained(model_name)
        mdl = T5ForConditionalGeneration.from_pretrained(model_name)
        tok.save_pretrained(str(tmp_dir))
        mdl.save_pretrained(str(tmp_dir))

        return cls(checkpoint_dir=str(tmp_dir), **kwargs)
