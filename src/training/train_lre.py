"""
Train the Linguistic Readiness Estimator (LRE) head.

The LRE head is a small MLP that sits on top of the frozen T5 encoder and
predicts a readiness score in [0, 1] — whether accumulated glosses are
complete enough to translate.

Training signal — Semantic Oracle:
  For each sentence, we translate every prefix length (1..n glosses) using
  the frozen T5 model, compute ROUGE-L against the reference, and use that
  score as the readiness label.  This teaches the LRE to predict actual
  translation quality rather than counting tokens.

  A monotonic constraint is enforced: readiness never decreases as more
  glosses arrive (more context can only help, never hurt).

  Labels are cached to disk after generation (~30 min on GPU) so that
  repeated LRE training runs reuse them instantly.

Run AFTER placing a fine-tuned T5 checkpoint in the checkpoint directory.

  python -m src.training.train_lre               # oracle labels (default)
  python -m src.training.train_lre --heuristic   # positional heuristic fallback
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.backends.t5_backend import LREHead
from src.config import cfg
from src.data.loader import load_aslg_pc12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Heuristic label generation (positional baseline) ─────────────────────────

def _readiness_label(position: int, total: int) -> float:
    """Soft label for partial gloss at `position` out of `total`."""
    frac = position / total
    if frac <= 1/3:
        return 0.0
    elif frac <= 2/3:
        return 0.7 * (frac - 1/3) / (1/3)
    elif frac < 1.0:
        return 0.7 + 0.25 * (frac - 2/3) / (1/3)
    else:
        return 1.0


def build_heuristic_examples(samples: List[Dict]) -> List[Dict]:
    """
    Build (partial_gloss, readiness_label) pairs using the positional heuristic.
    Each sentence contributes examples at all prefix lengths.
    """
    examples = []
    for s in samples:
        glosses = s["gloss"].split()
        n = len(glosses)
        if n < 2:
            continue
        for i in range(1, n + 1):
            partial = " ".join(glosses[:i])
            label   = _readiness_label(i, n)
            examples.append({
                "source": f"{cfg.t5.gloss_to_text_prefix}{partial}",
                "label":  label,
            })
    return examples


# ── Semantic Oracle label generation ─────────────────────────────────────────

def generate_oracle_labels(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    samples: List[Dict],
    device: torch.device,
    cache_path: Path = None,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Generate readiness labels by translating each gloss prefix with the
    frozen T5 and scoring the output against the full reference via ROUGE-L.

    A monotonic constraint ensures readiness(prefix_i) >= readiness(prefix_{i-1}):
    more context can only help, never hurt.

    Results are cached to ``cache_path`` for reuse across training runs.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached oracle labels from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    model.eval()
    examples = []

    for s in tqdm(samples, desc="Oracle labels", unit="sent"):
        glosses = s["gloss"].split()
        reference = s["text"]
        n = len(glosses)
        if n < 2:
            continue

        # Build all prefix source strings for this sentence
        sources = [
            f"{cfg.t5.gloss_to_text_prefix}{' '.join(glosses[:i])}"
            for i in range(1, n + 1)
        ]

        # Translate all prefixes (batch within this sentence)
        translations = []
        for start in range(0, len(sources), batch_size):
            batch = sources[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                max_length=cfg.t5.max_source_length,
                truncation=True,
                padding=True,
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=1,          # greedy — fast, labels don't need beam search
                )
            for seq in outputs:
                translations.append(
                    tokenizer.decode(seq, skip_special_tokens=True).strip()
                )

        # Compute ROUGE-L for each prefix translation vs. full reference
        scores = [
            scorer.score(reference, t)["rougeL"].fmeasure
            for t in translations
        ]

        # Monotonic constraint: readiness never decreases with more glosses
        for i in range(1, len(scores)):
            scores[i] = max(scores[i], scores[i - 1])

        for source, score in zip(sources, scores):
            examples.append({"source": source, "label": round(score, 4)})

    # Cache for reuse
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(examples, f)
        logger.info(f"Cached {len(examples)} oracle labels to {cache_path}")

    return examples


# Keep the public name that external code imports
build_lre_examples = build_heuristic_examples


# ── Dataset ───────────────────────────────────────────────────────────────────

class LREDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: T5TokenizerFast):
        self.examples  = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["source"],
            max_length=cfg.t5.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(ex["label"], dtype=torch.float32),
        }


# ── Main training function ────────────────────────────────────────────────────

def train_lre(
    checkpoint_dir: str = None,
    use_heuristic: bool = False,
) -> None:
    checkpoint_dir = Path(checkpoint_dir or cfg.t5.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"T5 checkpoint not found at {checkpoint_dir}. "
            "Place model files there or set t5.checkpoint_dir in config."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training LRE head on {device}")

    # ── Load T5 model ───────────────────────────────────────────────────────
    logger.info(f"Loading T5 from {checkpoint_dir}")
    tokenizer = T5TokenizerFast.from_pretrained(str(checkpoint_dir))
    model     = T5ForConditionalGeneration.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()

    # Freeze encoder for LRE training
    encoder = model.encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    hidden_dim = model.config.d_model

    # ── LRE head ────────────────────────────────────────────────────────────
    lre = LREHead(hidden_dim=hidden_dim).to(device)
    optimiser = torch.optim.AdamW(lre.parameters(), lr=cfg.training.lre_learning_rate)
    criterion = nn.MSELoss()

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("Loading training data for LRE...")
    train_samples, val_samples, _ = load_aslg_pc12(seed=cfg.training.random_seed)

    if use_heuristic:
        logger.info("Using positional heuristic labels.")
        train_examples = build_heuristic_examples(train_samples)
        val_examples   = build_heuristic_examples(val_samples)
    else:
        logger.info("Generating semantic oracle labels (cached after first run).")
        train_examples = generate_oracle_labels(
            model, tokenizer, train_samples, device,
            cache_path=checkpoint_dir / "oracle_labels_train.json",
        )
        val_examples = generate_oracle_labels(
            model, tokenizer, val_samples, device,
            cache_path=checkpoint_dir / "oracle_labels_val.json",
        )

    logger.info(f"LRE examples: train={len(train_examples)}, val={len(val_examples)}")

    train_loader = DataLoader(
        LREDataset(train_examples, tokenizer),
        batch_size=cfg.training.lre_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LREDataset(val_examples, tokenizer),
        batch_size=cfg.training.lre_batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(cfg.training.lre_epochs):
        # Train
        lre.train()
        train_loss = 0.0
        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            with torch.no_grad():
                enc_out = encoder(input_ids=ids, attention_mask=mask)

            pred = lre(enc_out.last_hidden_state, mask)
            loss = criterion(pred, label)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        lre.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                enc_out = encoder(input_ids=ids, attention_mask=mask)
                pred = lre(enc_out.last_hidden_state, mask)
                val_loss += criterion(pred, label).item()
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{cfg.training.lre_epochs} — "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in lre.state_dict().items()}

    # ── Save ────────────────────────────────────────────────────────────────
    if best_state:
        lre.load_state_dict(best_state)
    save_path = checkpoint_dir / "lre_head.pt"
    torch.save(lre.state_dict(), str(save_path))
    logger.info(f"LRE head saved to {save_path}  (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LRE head")
    parser.add_argument(
        "--heuristic", action="store_true",
        help="Use positional heuristic labels instead of semantic oracle labels",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to T5 checkpoint directory (default: from config)",
    )
    args = parser.parse_args()
    train_lre(checkpoint_dir=args.checkpoint_dir, use_heuristic=args.heuristic)
