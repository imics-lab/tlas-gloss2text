"""
Data loading for ASLG-PC12 (HuggingFace) and SIGNUM (local files).

Returns dicts with keys 'gloss' and 'text', already cleaned.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from datasets import load_dataset

from src.config import cfg

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove BOM, collapse whitespace, strip trailing ellipses."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "")
    text = re.sub(r"\.{2,}", "", text)
    return " ".join(text.split()).strip()


def clean_gloss(gloss: str) -> str:
    """Upper-case and clean a gloss sequence."""
    return clean_text(gloss).upper()


# ── ASLG-PC12 ────────────────────────────────────────────────────────────────

def load_aslg_pc12(
    num_train: int = None,
    num_val: int = None,
    num_test: int = None,
    seed: int = None,
    min_gloss_tokens: int = 3,
    min_text_tokens: int = 3,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load ASLG-PC12 from HuggingFace.

    Returns (train, val, test) lists of {'gloss': str, 'text': str}.
    """
    num_train = num_train or cfg.training.num_train_samples
    num_val   = num_val   or cfg.training.num_val_samples
    num_test  = num_test  or cfg.training.num_test_samples
    seed      = seed      or cfg.training.random_seed

    logger.info("Loading ASLG-PC12 from HuggingFace...")
    ds = load_dataset("achrafothman/aslg_pc12")
    full = ds["train"].shuffle(seed=seed)

    def _valid(item: Dict) -> bool:
        g = clean_gloss(item["gloss"])
        t = clean_text(item["text"]).lower()
        return (
            len(g.split()) >= min_gloss_tokens
            and len(t.split()) >= min_text_tokens
        )

    logger.info(f"Filtering {len(full)} samples (min {min_gloss_tokens} glosses)...")
    filtered = [
        {"gloss": clean_gloss(x["gloss"]), "text": clean_text(x["text"]).lower()}
        for x in full
        if _valid(x)
    ]
    logger.info(f"Kept {len(filtered)} samples after filtering.")

    n_needed = num_train + num_val + num_test
    if len(filtered) < n_needed:
        raise ValueError(
            f"Not enough samples: need {n_needed}, have {len(filtered)}"
        )

    train = filtered[:num_train]
    val   = filtered[num_train : num_train + num_val]
    test  = filtered[num_train + num_val : num_train + num_val + num_test]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ── SIGNUM ────────────────────────────────────────────────────────────────────

def load_signum(
    gloss_file: Path = None,
    trans_file: Path = None,
    min_gloss_tokens: int = 2,
    min_text_tokens: int = 2,
) -> List[Dict]:
    """
    Load the local SIGNUM parallel corpus.

    Format:
        signum_sents_anno_eng.txt  → "0001\tGLOSS GLOSS ..."
        signum_sents_trans_eng.txt → "0001\tEnglish text..."

    Returns list of {'gloss': str, 'text': str, 'id': str}.
    """
    gloss_file = gloss_file or cfg.paths.signum_glosses
    trans_file = trans_file or cfg.paths.signum_translations

    glosses = {}
    with open(gloss_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                glosses[parts[0]] = clean_gloss(parts[1])

    translations = {}
    with open(trans_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                translations[parts[0]] = clean_text(parts[1]).lower()

    samples = []
    for sid in sorted(glosses):
        if sid not in translations:
            continue
        g = glosses[sid]
        t = translations[sid]
        if len(g.split()) >= min_gloss_tokens and len(t.split()) >= min_text_tokens:
            samples.append({"id": sid, "gloss": g, "text": t})

    logger.info(f"Loaded {len(samples)} SIGNUM samples.")
    return samples


# ── Synthetic discourse loader ────────────────────────────────────────────────

def load_synthetic_discourse(
    path: Path = None,
    split: str = "train",
    test_size: int = 200,
) -> List[List[Dict]]:
    """
    Load generated discourse groups for context-pair training or evaluation.

    Args:
        path:      JSONL file path (defaults to data/synthetic_discourse.jsonl).
        split:     "train" (default), "test", or "all".
                   The first test_size groups are reserved as a held-out test set.
        test_size: Number of groups in the test split (default 200).

    Returns a list of sentence lists ready for make_context_pairs().
    """
    from src.training.synthetic_data import load_discourse_groups
    groups = load_discourse_groups(path, split=split, test_size=test_size)
    if groups:
        logger.info(f"Loaded {len(groups)} synthetic discourse groups "
                    f"({split} split, {sum(len(g) for g in groups)} sentences total).")
    return groups


# ── Combined loader ───────────────────────────────────────────────────────────

def load_all_training_data(
    include_signum: bool = True,
    seed: int = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load ASLG-PC12 (train/val/test) and optionally append SIGNUM to train.
    SIGNUM is only used for training augmentation, not evaluation.
    """
    seed = seed or cfg.training.random_seed
    train, val, test = load_aslg_pc12(seed=seed)

    if include_signum:
        signum = load_signum()
        train = train + signum
        rng = np.random.default_rng(seed)
        rng.shuffle(train)
        logger.info(f"After SIGNUM augmentation: train={len(train)}")

    return train, val, test
