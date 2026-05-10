"""
Discourse Context Evaluation — demonstrates that past context improves translation.

For each discourse group in the held-out test split, we translate each sentence:
  1. WITHOUT context: standard "translate ASL to English: GLOSS" input
  2. WITH context:    "translate ASL to English: [Context: prev_text] GLOSS"

We compare the BLEU/ROUGE scores, showing that context-aware translation is better.
This directly supports the paper's claim that prior context matters.

Additional analysis:
  - BLEU-by-position: shows that TLAS improves as conversation progresses

Run:
  python -m src.evaluation.evaluate_context [--n 200] [--backend t5]
"""

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import cfg
from src.data.loader import load_synthetic_discourse
from src.evaluation.metrics import compute_metrics, compute_bleu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _build_context_str(prior_texts: List[str]) -> str:
    return " ||| ".join(prior_texts)


async def evaluate_group(
    group: List[Dict],
    backend,
    context_window: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Translate all deaf turns in a discourse group twice:
      - without_ctx: each sentence translated in isolation
      - with_ctx:    each sentence translated with prior translations as context

    Returns (without_ctx_results, with_ctx_results), each a list of:
      {"gloss": str, "reference": str, "hypothesis": str, "position": int}
    """
    from src.backends.base import Direction  # noqa: F401 — used via backend.translate()

    without_ctx: List[Dict] = []
    with_ctx: List[Dict] = []

    # Collect translations as we go (for context construction)
    prior_translations: List[str] = []  # running translations (with-context version)
    deaf_position = 0  # track position of deaf turns within the group

    for turn in group:
        if "gloss" not in turn:
            # Hearing turn — add its text to the context window and skip
            prior_translations.append(turn["text"])
            continue

        gloss     = turn["gloss"]
        reference = turn["text"]

        # 1. Without context — translate in isolation
        result_no_ctx = await backend.translate(gloss, direction=Direction.GLOSS_TO_TEXT)
        hyp_no_ctx    = result_no_ctx.translation

        # 2. With context — prepend prior translations
        ctx_window = prior_translations[-context_window:]
        if ctx_window:
            ctx_str    = _build_context_str(ctx_window)
            result_ctx = await backend.translate(
                gloss, context=ctx_str, direction=Direction.GLOSS_TO_TEXT
            )
            hyp_ctx = result_ctx.translation
        else:
            hyp_ctx = hyp_no_ctx  # first turn — no prior context yet

        without_ctx.append({"gloss": gloss, "reference": reference, "hypothesis": hyp_no_ctx, "position": deaf_position})
        with_ctx.append(   {"gloss": gloss, "reference": reference, "hypothesis": hyp_ctx, "position": deaf_position})

        # Advance context using the with-context hypothesis
        prior_translations.append(hyp_ctx)
        deaf_position += 1

    return without_ctx, with_ctx


async def run_context_evaluation(
    backend_name: str = "t5",
    n_groups: Optional[int] = None,
    test_size: int = 200,
    context_window: int = 3,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Run the context ablation evaluation on the discourse test split.

    Returns a dict with keys "without_context" and "with_context", each
    containing aggregated metrics (BLEU, ROUGE-L).
    """
    output_path = output_path or (
        Path(cfg.evaluation.results_dir) / f"context_eval_{backend_name}.json"
    )

    # Load test split
    groups = load_synthetic_discourse(split="test", test_size=test_size)
    if n_groups:
        groups = groups[:n_groups]
    logger.info(f"Evaluating context benefit on {len(groups)} discourse groups "
                f"({backend_name} backend)")

    # Load backend
    backend = _load_backend(backend_name)

    all_no_ctx: List[Dict] = []
    all_with_ctx: List[Dict] = []

    for i, group in enumerate(groups):
        no_ctx, w_ctx = await evaluate_group(group, backend, context_window)
        all_no_ctx.extend(no_ctx)
        all_with_ctx.extend(w_ctx)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(groups)} groups processed")

    # Compute metrics
    refs_no_ctx  = [r["reference"]  for r in all_no_ctx]
    hyps_no_ctx  = [r["hypothesis"] for r in all_no_ctx]
    refs_with_ctx = [r["reference"]  for r in all_with_ctx]
    hyps_with_ctx = [r["hypothesis"] for r in all_with_ctx]

    metrics_no_ctx   = compute_metrics(hyps_no_ctx,  refs_no_ctx,  compute_bertscore=False)
    metrics_with_ctx = compute_metrics(hyps_with_ctx, refs_with_ctx, compute_bertscore=False)

    # BLEU by position within discourse group
    bleu_by_pos_no_ctx = _compute_bleu_by_position(all_no_ctx)
    bleu_by_pos_with_ctx = _compute_bleu_by_position(all_with_ctx)

    results = {
        "backend":        backend_name,
        "n_groups":       len(groups),
        "n_sentences":    len(all_no_ctx),
        "context_window": context_window,
        "without_context": {
            "bleu":   metrics_no_ctx.bleu,
            "rouge_l": metrics_no_ctx.rouge_l,
        },
        "with_context": {
            "bleu":   metrics_with_ctx.bleu,
            "rouge_l": metrics_with_ctx.rouge_l,
        },
        "delta": {
            "bleu":   round(metrics_with_ctx.bleu   - metrics_no_ctx.bleu,   2),
            "rouge_l": round(metrics_with_ctx.rouge_l - metrics_no_ctx.rouge_l, 4),
        },
        "bleu_by_position": {
            "without_context": bleu_by_pos_no_ctx,
            "with_context":    bleu_by_pos_with_ctx,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_results(results)
    logger.info(f"Results saved to {output_path}")
    await backend.close()
    return results


def _compute_bleu_by_position(results: List[Dict]) -> Dict[str, float]:
    """
    Group results by their position within the discourse group, then compute
    BLEU for each position bucket.

    Returns {"0": bleu, "1": bleu, ...} for positions 0–4+.
    """
    by_pos = defaultdict(lambda: {"hyps": [], "refs": []})
    for r in results:
        pos = min(r["position"], 4)  # bucket 4+ together
        by_pos[pos]["hyps"].append(r["hypothesis"])
        by_pos[pos]["refs"].append(r["reference"])

    bleu_by_pos = {}
    for pos in sorted(by_pos.keys()):
        bucket = by_pos[pos]
        bleu_score = compute_bleu(bucket["hyps"], bucket["refs"])
        label = f"{pos}+" if pos == 4 else str(pos)
        bleu_by_pos[label] = round(bleu_score, 2)
    return bleu_by_pos


def _print_results(r: Dict) -> None:
    nc = r["without_context"]
    wc = r["with_context"]
    d  = r["delta"]
    sign = lambda v: f"+{v:.2f}" if v >= 0 else f"{v:.2f}"
    print(f"\n{'='*55}")
    print(f"Context Benefit Evaluation — {r['backend']} backend")
    print(f"  Groups: {r['n_groups']}   Sentences: {r['n_sentences']}")
    print(f"  Context window: {r['context_window']} prior turns")
    print(f"{'='*55}")
    print(f"{'Metric':<15} {'Without ctx':>12} {'With ctx':>12} {'Delta':>10}")
    print(f"{'-'*55}")
    print(f"{'BLEU':<15} {nc['bleu']:>12.2f} {wc['bleu']:>12.2f} {sign(d['bleu']):>10}")
    print(f"{'ROUGE-L':<15} {nc['rouge_l']:>12.4f} {wc['rouge_l']:>12.4f} {sign(d['rouge_l']):>10}")
    print(f"{'='*55}")

    # Position-based BLEU
    bp = r.get("bleu_by_position")
    if bp:
        nc_bp = bp.get("without_context", {})
        wc_bp = bp.get("with_context", {})
        print(f"\nBLEU by position within discourse group:")
        print(f"{'Position':<12} {'Without ctx':>12} {'With ctx':>12} {'Delta':>10}")
        print(f"{'-'*48}")
        for pos in sorted(set(list(nc_bp.keys()) + list(wc_bp.keys()))):
            nc_v = nc_bp.get(pos, 0.0)
            wc_v = wc_bp.get(pos, 0.0)
            delta = wc_v - nc_v
            print(f"{pos:<12} {nc_v:>12.2f} {wc_v:>12.2f} {sign(delta):>10}")
    print()


def _load_backend(name: str):
    name = name.lower()
    if name == "t5":
        from src.backends.t5_backend import T5Backend
        return T5Backend()
    elif name == "gemini":
        from src.backends.gemini_backend import GeminiBackend
        return GeminiBackend()
    elif name == "groq":
        from src.backends.groq_backend import GroqBackend
        return GroqBackend()
    elif name in ("ollama", "gpt_oss"):
        from src.backends.ollama_backend import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the benefit of discourse context on translation quality."
    )
    p.add_argument("--backend",  type=str, default="t5",
                   choices=["t5", "gemini", "groq", "ollama"])
    p.add_argument("--n",        type=int, default=None,
                   help="Number of test groups to evaluate (default: all 200)")
    p.add_argument("--test-size", type=int, default=200,
                   help="Size of the held-out test split")
    p.add_argument("--window",   type=int, default=3,
                   help="Context window size (number of prior turns)")
    p.add_argument("--output",   type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out  = Path(args.output) if args.output else None
    asyncio.run(run_context_evaluation(
        backend_name=args.backend,
        n_groups=args.n,
        test_size=args.test_size,
        context_window=args.window,
        output_path=out,
    ))
