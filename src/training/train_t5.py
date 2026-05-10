"""
T5 fine-tuning for bidirectional gloss↔text translation.

Training data:
  1. Standard seq2seq pairs: full gloss → full English text  (gloss→text)
  2. Reverse pairs:          full English text → full gloss  (text→gloss)
  3. TransLLaMa-style:       partial gloss → <WAIT> / partial text / full text
  4. Multi-sentence context: [Context: prev] full gloss → full text

Run:
  python -m src.training.train_t5
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.config import cfg
from src.data.loader import load_all_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Training pair generation ──────────────────────────────────────────────────

def make_standard_pair(gloss: str, text: str) -> List[Dict]:
    """Standard gloss→text and text→gloss pairs."""
    return [
        {
            "source": f"{cfg.t5.gloss_to_text_prefix}{gloss}",
            "target": text,
        },
        {
            "source": f"{cfg.t5.text_to_gloss_prefix}{text}",
            "target": gloss,
        },
    ]


def make_transllama_pairs(gloss: str, text: str) -> List[Dict]:
    """
    Create TransLLaMa-style training examples at strategic positions:
      - Early (first 1/3): target = <WAIT>
      - Mid   (first 1/2): target = partial translation
      - Full:              target = complete translation  (already in standard pairs)
    """
    glosses = gloss.split()
    words   = text.split()
    n = len(glosses)
    if n < 3:
        return []

    pairs = []

    # Early point — not enough context yet
    early = max(1, n // 3)
    pairs.append({
        "source": f"{cfg.t5.gloss_to_text_prefix}{' '.join(glosses[:early])}",
        "target": cfg.t5.wait_token,
    })

    # Mid point — can provide partial translation
    mid = n // 2
    if mid > early:
        partial_words = words[:max(1, len(words) // 2)]
        pairs.append({
            "source": f"{cfg.t5.gloss_to_text_prefix}{' '.join(glosses[:mid])}",
            "target": " ".join(partial_words) if partial_words else cfg.t5.wait_token,
        })

    return pairs[:cfg.training.max_streaming_examples_per_sentence - 1]


def make_context_pairs(
    samples: List[Dict],
    window: int = 3,
) -> List[Dict]:
    """
    Multi-sentence context pairs.  Prepend previous translations as context.
    Teaches the model to use cross-sentence information.

    Handles two turn types:
      {"gloss": str, "text": str}  – deaf turn: translation target + context
      {"text": str}                – hearing turn: contributes to context only

    All turns (deaf and hearing) contribute their "text" to the context window.
    Only deaf turns (with "gloss") become translation targets.
    The first deaf turn in a group is skipped if it has no prior context.
    """
    pairs = []
    for i, sample in enumerate(samples):
        if "gloss" not in sample:
            continue  # hearing turn — context only, not a translation target
        ctx_start = max(0, i - window)
        context = " ||| ".join(s["text"] for s in samples[ctx_start:i])
        if not context:
            continue  # skip the very first deaf turn (no prior context yet)
        pairs.append({
            "source": f"{cfg.t5.gloss_to_text_prefix}[Context: {context}] {sample['gloss']}",
            "target": sample["text"],
        })
    return pairs


def build_training_pairs(
    samples: List[Dict],
    discourse_groups: List[List[Dict]] = None,
) -> List[Dict]:
    """
    Build the complete set of training pairs from a list of samples.

    Args:
        samples:          Flat list of {'gloss', 'text'} dicts (ASLG-PC12 + SIGNUM).
        discourse_groups: Optional list of discourse groups from synthetic_data.
                          Each group is a connected sentence list; context pairs
                          are extracted using a sliding window over each group.
    """
    pairs = []
    for s in samples:
        pairs.extend(make_standard_pair(s["gloss"], s["text"]))
        pairs.extend(make_transllama_pairs(s["gloss"], s["text"]))
    # Pseudo-discourse context pairs from the flat sample list
    pairs.extend(make_context_pairs(samples, window=cfg.tlas.context_window_size))
    # Genuine discourse context pairs from synthetic connected groups
    if discourse_groups:
        for group in discourse_groups:
            pairs.extend(make_context_pairs(group, window=cfg.tlas.context_window_size))
        n_ctx = sum(max(0, len(g) - 1) for g in discourse_groups)
        logger.info(f"Added {n_ctx} genuine discourse context pairs from "
                    f"{len(discourse_groups)} synthetic groups.")
    logger.info(f"Built {len(pairs)} training pairs from {len(samples)} samples.")
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────

class Seq2SeqDataset(Dataset):
    def __init__(self, pairs: List[Dict], tokenizer: T5TokenizerFast):
        self.pairs     = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        source_enc = self.tokenizer(
            pair["source"],
            max_length=cfg.t5.max_source_length,
            truncation=True,
            padding=False,
        )
        target_enc = self.tokenizer(
            pair["target"],
            max_length=cfg.t5.max_target_length,
            truncation=True,
            padding=False,
        )
        labels = target_enc["input_ids"]
        # Replace pad token id with -100 so it's ignored in the loss
        labels = [
            l if l != self.tokenizer.pad_token_id else -100
            for l in labels
        ]
        return {
            "input_ids":      source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels":         labels,
        }


# ── Main training function ────────────────────────────────────────────────────

def train(
    output_dir: str = None,
    resume_from_checkpoint: bool = True,
    include_synthetic_discourse: bool = False,
) -> None:
    output_dir = output_dir or cfg.t5.checkpoint_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_samples, val_samples, _ = load_all_training_data(
        include_signum=True, seed=cfg.training.random_seed
    )

    discourse_groups = None
    if include_synthetic_discourse:
        from src.data.loader import load_synthetic_discourse
        discourse_groups = load_synthetic_discourse()
        if not discourse_groups:
            logger.warning(
                "No synthetic discourse found. "
                "Run: python -m src.training.synthetic_data --mode discourse"
            )

    # ── Build training pairs ───────────────────────────────────────────────────
    train_pairs = build_training_pairs(train_samples, discourse_groups=discourse_groups)
    val_pairs   = build_training_pairs(val_samples)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {cfg.t5.model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(cfg.t5.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [cfg.t5.wait_token]}
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {cfg.t5.model_name}")
    model = T5ForConditionalGeneration.from_pretrained(cfg.t5.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = Seq2SeqDataset(train_pairs, tokenizer)
    val_ds   = Seq2SeqDataset(val_pairs,   tokenizer)
    logger.info(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    # ── Training args ──────────────────────────────────────────────────────────
    tc = cfg.training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc.num_epochs,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        warmup_ratio=tc.warmup_ratio,
        weight_decay=tc.weight_decay,
        label_smoothing_factor=tc.label_smoothing_factor,
        fp16=tc.fp16 and torch.cuda.is_available(),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=tc.random_seed,
        dataloader_num_workers=2,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    has_checkpoint = resume_from_checkpoint and any(Path(output_dir).glob("checkpoint-*"))
    checkpoint = output_dir if has_checkpoint else None
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model and tokenizer ────────────────────────────────────────
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune T5 for gloss↔text translation.")
    p.add_argument("--output-dir",  type=str, default=None)
    p.add_argument("--no-resume",   action="store_true", help="Start fresh, ignore checkpoint")
    p.add_argument("--discourse",   action="store_true",
                   help="Include synthetic discourse context pairs (requires data/synthetic_discourse.jsonl)")
    args = p.parse_args()
    train(
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume,
        include_synthetic_discourse=args.discourse,
    )
