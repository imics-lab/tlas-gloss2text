"""
Full evaluation matrix: all (backend × policy) combinations.

Datasets:
  - aslg:      ASLG-PC12 test split (isolated sentences, synthetic timestamps)
  - signum:    SIGNUM corpus (isolated sentences, synthetic timestamps)
  - discourse: Synthetic discourse test set (connected groups with LLM timestamps)

Run:
  python -m src.evaluation.evaluate [--backend t5] [--policies all] [--n 100]
  python -m src.evaluation.evaluate --dataset discourse --backend t5 --n 50
"""

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from src.config import cfg
from src.data.loader import load_aslg_pc12
from src.evaluation.metrics import (
    MetricResult,
    compute_all_metrics,
    compute_bleu,
    format_metrics_table,
)
from src.pipeline import StreamingPipeline, SentenceResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Backend factory ───────────────────────────────────────────────────────────

def build_backend(name: str):
    name = name.lower()
    if name == "t5":
        from src.backends.t5_backend import T5Backend
        return T5Backend()
    elif name == "groq":
        from src.backends.groq_backend import GroqBackend
        return GroqBackend()
    elif name == "gemini":
        from src.backends.gemini_backend import GeminiBackend
        return GeminiBackend()
    elif name in ("ollama", "gpt_oss"):
        from src.backends.ollama_backend import OllamaBackend
        return OllamaBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")


# ── Canonical policy ordering for result tables ────────────────────────────────
# Groups: oracle | external baselines | our ablations (LSG, TLAS variants)
# This ordering is used by both the Markdown formatter and the console printer.
POLICY_ORDER = [
    # Upper bound
    "Batch (oracle)",
    # External baselines (published methods)
    "Batch",
    "Wait-k",
    "TransLLaMa",
    # Ablations — this work (not external baselines)
    "LSG",
    "TLAS-linguistic",
    "TLAS-temporal",
    "TLAS",
]

# Names that are our own work (ablations), not external published baselines
OUR_ABLATIONS = {"LSG", "TLAS", "TLAS-temporal", "TLAS-linguistic"}


# ── Policy factory ────────────────────────────────────────────────────────────

def build_policies(backend, names: Optional[List[str]] = None):
    """
    Return a {name: policy} dict in canonical display order.
    If names is None or ["all"], returns every policy.
    """
    from src.policy.baselines import (
        BatchPolicy, WaitKPolicy, TransLLaMaPolicy, LSGPolicy,
    )
    from src.policy.tlas import TLASPolicy, TLASMode

    all_policies = {
        # External baselines (published methods)
        "Batch":            BatchPolicy(),
        "Wait-k":           WaitKPolicy(),
        "TransLLaMa":       TransLLaMaPolicy(backend),
        # Ablations — this work
        "LSG":              LSGPolicy(backend),
        "TLAS-linguistic":  TLASPolicy(backend, mode=TLASMode.LINGUISTIC_ONLY),
        "TLAS-temporal":    TLASPolicy(backend, mode=TLASMode.TEMPORAL_ONLY),
        "TLAS":             TLASPolicy(backend, mode=TLASMode.FULL),
    }

    if names is None or names == ["all"]:
        return all_policies

    return {k: v for k, v in all_policies.items() if k in names}


# ── Single policy evaluation ──────────────────────────────────────────────────

async def evaluate_policy(
    backend,
    policy,
    test_samples: List[Dict],
    compute_bertscore: bool = True,
    context_window: int = None,
) -> MetricResult:
    """
    Run one (backend, policy) pair over test_samples.
    Returns MetricResult.
    """
    pipeline = StreamingPipeline(backend, policy, verbose=False,
                                 context_window=context_window)
    sentence_results: List[SentenceResult] = await pipeline.run_sentence_list(
        test_samples, speed_multiplier=0.0
    )
    metrics = compute_all_metrics(
        sentence_results,
        batch_bleu=None,  # filled in later for retention
        compute_bertscore_flag=compute_bertscore,
        bertscore_model=cfg.evaluation.bertscore_model,
    )
    return metrics, sentence_results


# ── Full matrix evaluation ────────────────────────────────────────────────────

async def run_evaluation(
    backend_name: str = "t5",
    policy_names: Optional[List[str]] = None,
    n_samples: int = 100,
    compute_bertscore: bool = True,
    output_dir: Optional[str] = None,
    dataset: str = "aslg",
) -> Dict[str, MetricResult]:
    """
    Evaluate all requested policies on `n_samples` test examples.

    Args:
        backend_name:     Which backend to use.
        policy_names:     List of policy names, or None/"all" for everything.
        n_samples:        Number of test samples.
        compute_bertscore: Whether to run BERTScore (slow, ~30s on GPU).
        output_dir:       Where to save results JSON + table.
        dataset:          "aslg" (default) or "signum".

    Returns:
        {policy_name: MetricResult}
    """
    output_dir = Path(output_dir or cfg.evaluation.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    if dataset == "signum":
        from src.data.loader import load_signum
        test_samples = load_signum()
        if n_samples and n_samples < len(test_samples):
            test_samples = test_samples[:n_samples]
    else:
        _, _, test_samples = load_aslg_pc12(seed=cfg.training.random_seed)
        test_samples = test_samples[:n_samples]
    logger.info(f"Evaluating on {len(test_samples)} test samples (dataset={dataset}).")

    # ── Build backend ─────────────────────────────────────────────────────────
    logger.info(f"Loading backend: {backend_name}")
    backend = build_backend(backend_name)

    # T5 was not fine-tuned with discourse context; disable context window to
    # prevent context contamination (LLM backends handle context via prompting).
    ctx_window = 0 if backend_name == "t5" else None

    # ── Build policies ────────────────────────────────────────────────────────
    policies = build_policies(backend, policy_names)
    logger.info(f"Policies to evaluate: {list(policies.keys())}")

    # ── Evaluate each policy (save after each one) ───────────────────────────
    all_results: Dict[str, List[SentenceResult]] = {}
    suffix = f"_{dataset}" if dataset != "aslg" else ""
    results_file = output_dir / f"eval_{backend_name}{suffix}.json"
    table_file   = output_dir / f"eval_{backend_name}{suffix}_table.md"

    # Load previously saved metrics so re-runs accumulate rather than overwrite.
    all_metrics: Dict[str, MetricResult] = {}
    if results_file.exists():
        try:
            with results_file.open("r", encoding="utf-8") as f:
                prior = json.load(f)
            for pol_name, d in prior.get("metrics", {}).items():
                m = MetricResult(**{k: v for k, v in d.items() if k != "extras"})
                all_metrics[pol_name] = m
            logger.info(f"Loaded {len(all_metrics)} existing results from {results_file}")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    def _save():
        """Persist current results to disk (called after every policy)."""
        batch_bleu = all_metrics.get("Batch", MetricResult()).bleu
        if batch_bleu > 0:
            for m in all_metrics.values():
                m.retention = m.bleu / batch_bleu
        data = {
            "backend": backend_name,
            "n_samples": n_samples,
            "metrics": {k: v.to_dict() for k, v in all_metrics.items()},
        }
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        table = format_metrics_table(all_metrics)
        table_file.write_text(f"# Evaluation: {backend_name}\n\n{table}\n", encoding="utf-8")

    for pol_name, policy in policies.items():
        logger.info(f"Evaluating: {pol_name} ...")
        metrics, sent_results = await evaluate_policy(
            backend, policy, test_samples, compute_bertscore=compute_bertscore,
            context_window=ctx_window,
        )
        all_metrics[pol_name] = metrics
        all_results[pol_name] = sent_results
        bs_str = f"{metrics.bertscore_f1:.4f}" if metrics.bertscore_f1 is not None else "N/A"
        logger.info(
            f"  {pol_name}: BLEU={metrics.bleu:.2f}  "
            f"ROUGE-L={metrics.rougeL:.4f}  "
            f"BS={bs_str}  "
            f"AL={metrics.avg_lagging:.3f}"
        )
        _save()  # write to disk immediately after each policy

    # ── Print final Markdown table ────────────────────────────────────────────
    table = format_metrics_table(all_metrics)
    logger.info(f"Results saved to {results_file}")
    print(f"\n## Results — backend: {backend_name}\n")
    print(table)

    await backend.close()
    return all_metrics


# ── Discourse evaluation (continuous stream with LLM timestamps) ─────────────
#
# Each discourse group is treated as a continuous gloss stream — the policy
# is NOT told where sentence boundaries are.  It must discover them using
# its own signals (temporal for TLAS, linguistic for TransLLaMa/LSG, fixed
# windowing for Wait-k, etc.).
#
# Two WRITE mechanisms for TLAS policies:
#   A) Proactive timeout: before processing a new gloss, if the gap since the
#      last gloss exceeds an adaptive threshold (TPD EMA × pause_multiplier),
#      translate the current buffer *before* adding the new gloss.  This
#      simulates the real system's timer that fires during silence.
#   B) Reactive decision: after processing a gloss, the policy's combined
#      TPD + LRE signal exceeds the AFG threshold → WRITE.
#
# Batch with oracle boundaries is evaluated separately as an upper bound.
# ─────────────────────────────────────────────────────────────────────────────


def _has_adaptive_timeout(policy) -> bool:
    """Check if policy supports proactive gap-based timeout.

    Enabled for TLAS-full and TLAS-temporal, but NOT TLAS-linguistic.
    The linguistic ablation must rely solely on the LRE signal (+ max_lag)
    to provide a clean measurement of linguistic readiness alone.
    """
    if not (hasattr(policy, 'tpd') and hasattr(policy.tpd, 'current_ema')):
        return False
    # Exclude LINGUISTIC_ONLY mode from proactive timeout
    from src.policy.tlas import TLASMode
    if hasattr(policy, 'mode') and policy.mode == TLASMode.LINGUISTIC_ONLY:
        return False
    return True


def _build_event_stream(group: List[Dict], timestamp_mode: str = "realistic"):
    """
    Flatten a discourse group into a continuous event stream.

    Args:
        group:          List of sentence dicts from the discourse JSONL.
        timestamp_mode: Controls temporal signal quality:
            "realistic" — use LLM-generated per-gloss timestamps (default)
            "uniform"   — constant 450ms gaps (destroys temporal signal)
            "noisy"     — add Gaussian jitter to realistic timestamps
                          (simulates vision module latency)

    Returns:
        events:     [(timestamp_sec, gloss_token, deaf_sentence_idx), ...]
        references: {deaf_sentence_idx: reference_text}
        hearing:    [(timestamp_sec, text), ...] — hearing turns for context
    """
    import numpy as np
    rng = np.random.default_rng(seed=42)

    events = []
    references = {}
    hearing = []
    deaf_idx = 0

    for sent in group:
        if "gloss" not in sent:
            # Hearing turn — context only (no gloss events)
            ts = sent.get("timestamp_ms", 0) / 1000.0
            hearing.append((ts, sent["text"]))
            continue

        glosses = sent["gloss"].split()
        ts_ms = sent.get("gloss_timestamps_ms")

        if timestamp_mode == "uniform":
            # Constant 450ms gaps — no inter-sentence pause signal
            start = events[-1][0] + 0.45 if events else 0.0
            timestamps = [start + i * 0.45 for i in range(len(glosses))]
        elif timestamp_mode == "noisy":
            # Start from realistic timestamps, then add heavy jitter
            if ts_ms and len(ts_ms) == len(glosses):
                timestamps = [t / 1000.0 for t in ts_ms]
            else:
                start = events[-1][0] + 3.0 if events else 0.0
                timestamps = [start + i * 0.45 for i in range(len(glosses))]
            # Jitter std = 500ms — large enough to blur inter-sentence gaps
            timestamps = [t + rng.normal(0, 0.5) for t in timestamps]
            # Enforce monotonicity (jitter can cause inversions)
            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i - 1]:
                    timestamps[i] = timestamps[i - 1] + 0.05
        else:
            # "realistic" — use LLM-generated timestamps
            if ts_ms and len(ts_ms) == len(glosses):
                timestamps = [t / 1000.0 for t in ts_ms]
            else:
                start = events[-1][0] + 3.0 if events else 0.0
                timestamps = [start + i * 0.45 for i in range(len(glosses))]

        for g, t in zip(glosses, timestamps):
            events.append((t, g, deaf_idx))
        references[deaf_idx] = sent["text"]
        deaf_idx += 1

    return events, references, hearing


def _align_segments_to_references(
    segments: List[tuple],
    references: Dict[int, str],
) -> List[Dict]:
    """
    Align policy-produced segments to gold reference sentences.

    Each segment is (translation_text, [sentence_indices_of_its_glosses]).
    A segment is assigned to the reference sentence that contributes the
    majority of its glosses (ties broken by first occurrence).

    Returns a list of {"hypothesis", "reference", "position"} dicts,
    one per reference sentence.
    """
    from collections import Counter

    hyps_by_ref = defaultdict(list)
    for hyp_text, sent_indices in segments:
        if not sent_indices:
            continue
        # Majority vote: which reference sentence owns most glosses?
        counter = Counter(sent_indices)
        primary_ref = counter.most_common(1)[0][0]
        hyps_by_ref[primary_ref].append(hyp_text)

    pairs = []
    for ref_idx in sorted(references.keys()):
        ref_text = references[ref_idx]
        hyp_parts = hyps_by_ref.get(ref_idx, [""])
        hyp_text = " ".join(h for h in hyp_parts if h.strip())
        pairs.append({
            "hypothesis": hyp_text or "",
            "reference": ref_text,
            "position": ref_idx,
        })
    return pairs


async def _evaluate_group_stream(
    backend,
    policy,
    group: List[Dict],
    context_window: int = 3,
    timestamp_mode: str = "realistic",
) -> List[Dict]:
    """
    Evaluate one policy on one discourse group as a continuous stream.

    The policy receives glosses one-by-one with realistic timestamps and
    must decide when to WRITE (segment and translate).  The policy is reset
    ONCE at the start of the group, NOT between sentences.

    For TLAS policies, a proactive timeout fires when the gap since the last
    gloss exceeds TPD's adaptive threshold (EMA × pause_multiplier), causing
    a WRITE *before* the next gloss arrives.  This simulates the real-time
    system's idle-timeout.

    Returns [(hypothesis, reference, position), ...] per reference sentence.
    """
    from src.policy.afg import PolicyDecision, AFGDecision

    events, references, hearing = _build_event_stream(group, timestamp_mode)
    if not events:
        return []

    policy.reset()  # ONCE per group
    segments = []   # [(translation, [sentence_indices])]
    context = []    # sliding window of prior translations
    sent_tracker = []   # sentence idx for each gloss in current buffer
    hearing_idx = 0
    adaptive = _has_adaptive_timeout(policy)

    async def _translate_and_record():
        """Translate current buffer and record the segment."""
        nonlocal sent_tracker
        buf = list(policy.buffer)
        if not buf:
            return
        ctx_str = " ||| ".join(context[-context_window:]) if (context and context_window > 0) else ""
        try:
            result = await backend.translate(
                " ".join(buf), context=ctx_str,
            )
            hyp = result.translation.strip()
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            hyp = ""
        segments.append((hyp, list(sent_tracker)))
        if hyp:
            context.append(hyp)
            if len(context) > context_window:
                context.pop(0)
        sent_tracker = []

    for i, (ts, gloss, sent_idx) in enumerate(events):

        # ── Inject hearing context at appropriate timestamps ──
        while hearing_idx < len(hearing):
            h_ts, h_text = hearing[hearing_idx]
            if h_ts <= ts:
                context.append(h_text)
                if len(context) > context_window:
                    context.pop(0)
                hearing_idx += 1
            else:
                break

        # ── Scenario A: Proactive timeout (TLAS only) ──
        # If the gap since the last gloss exceeds the adaptive threshold,
        # translate the current buffer BEFORE adding the new gloss.
        if adaptive and policy.buffer and policy.tpd._state.last_timestamp is not None:
            gap = ts - policy.tpd._state.last_timestamp
            adaptive_timeout = policy.tpd.current_ema * policy.tpd.pause_multiplier
            if gap > adaptive_timeout:
                await _translate_and_record()
                policy.flush()
                # Reset last_timestamp so TPD doesn't double-fire on the gap
                # when update() is called, but preserve the EMA for adaptation.
                policy.tpd._state.last_timestamp = None

        # ── Feed gloss to policy ──
        sent_tracker.append(sent_idx)
        is_final = (i == len(events) - 1)
        decision = await policy.step(gloss, ts, is_final=is_final)

        # ── Scenario B: Reactive policy decision ──
        if decision.decision == PolicyDecision.WRITE:
            await _translate_and_record()
            policy.flush()

    # ── Flush any remaining buffer (safety, should be empty after is_final) ──
    if policy.buffer:
        await _translate_and_record()
        policy.flush()

    return _align_segments_to_references(segments, references)


async def _evaluate_group_oracle(
    backend,
    group: List[Dict],
    context_window: int = 3,
) -> List[Dict]:
    """
    Evaluate Batch policy with oracle sentence boundaries (upper bound).

    Each sentence is translated independently with full discourse context
    accumulated from prior translations.  This represents the best possible
    segmentation — a ceiling that streaming policies try to approach.
    """
    context = []
    results = []
    deaf_position = 0

    for sent in group:
        if "gloss" not in sent:
            # Hearing turn — advance context
            context.append(sent["text"])
            if len(context) > context_window:
                context.pop(0)
            continue

        gloss = sent["gloss"]
        reference = sent["text"]
        ctx_str = " ||| ".join(context[-context_window:]) if context else ""

        try:
            result = await backend.translate(gloss, context=ctx_str)
            hyp = result.translation.strip()
        except Exception as e:
            logger.warning(f"Oracle translation failed: {e}")
            hyp = ""

        results.append({
            "hypothesis": hyp,
            "reference": reference,
            "position": deaf_position,
        })
        deaf_position += 1

        # Update context with the translation
        if hyp:
            context.append(hyp)
            if len(context) > context_window:
                context.pop(0)

    return results


async def evaluate_policy_on_discourse(
    backend,
    policy,
    groups: List[List[Dict]],
    context_window: int = 3,
    compute_bertscore: bool = False,
    compute_sbert: bool = False,
    compute_chrf: bool = False,
    oracle_boundaries: bool = False,
    timestamp_mode: str = "realistic",
) -> Dict:
    """
    Run one policy on a list of discourse groups.

    If oracle_boundaries=True: per-sentence Batch evaluation (upper bound).
    If oracle_boundaries=False: continuous stream evaluation (realistic).

    Returns a dict with overall metrics + per-position BLEU breakdown.
    """
    from src.evaluation.metrics import compute_bleu, compute_metrics

    all_results = []

    try:
        from tqdm import tqdm
        label = "Batch (oracle)" if oracle_boundaries else policy.name
        group_iter = tqdm(groups, desc=label, unit="group", leave=False)
    except ImportError:
        group_iter = groups

    for group in group_iter:
        if oracle_boundaries:
            pairs = await _evaluate_group_oracle(backend, group, context_window)
        else:
            pairs = await _evaluate_group_stream(
                backend, policy, group, context_window,
                timestamp_mode=timestamp_mode,
            )
        all_results.extend(pairs)

    # Compute overall metrics
    hyps = [r["hypothesis"] for r in all_results]
    refs = [r["reference"] for r in all_results]
    srcs = [r.get("source", r["reference"]) for r in all_results]  # gloss or ref as fallback
    overall = compute_metrics(
        hyps, refs, sources=srcs,
        compute_bertscore=compute_bertscore,
        compute_sbert=compute_sbert,
        compute_chrf=compute_chrf,
    )

    # Compute BLEU by position
    by_pos = defaultdict(lambda: {"hyps": [], "refs": []})
    for r in all_results:
        pos = min(r["position"], 4)
        by_pos[pos]["hyps"].append(r["hypothesis"])
        by_pos[pos]["refs"].append(r["reference"])

    bleu_by_pos = {}
    for pos in sorted(by_pos.keys()):
        bucket = by_pos[pos]
        bleu_score = compute_bleu(bucket["hyps"], bucket["refs"])
        label = f"{pos}+" if pos == 4 else str(pos)
        bleu_by_pos[label] = round(bleu_score, 2)

    return {
        "bleu": overall.bleu,
        "rouge_l": overall.rouge_l,
        "bertscore_f1": overall.bertscore_f1,
        "sbert": overall.sbert,
        "chrf": overall.chrf,
        "num_sentences": len(all_results),
        "bleu_by_position": bleu_by_pos,
    }


async def run_discourse_evaluation(
    backend_name: str = "t5",
    policy_names: Optional[List[str]] = None,
    n_groups: Optional[int] = None,
    test_size: int = 200,
    compute_bertscore: bool = False,
    compute_sbert: bool = False,
    compute_chrf: bool = False,
    output_dir: Optional[str] = None,
    timestamp_mode: str = "realistic",
) -> Dict:
    """
    Evaluate all policies on the synthetic discourse test set.

    Each discourse group is treated as a continuous gloss stream.  Policies
    must discover sentence boundaries using their own signals.  Batch with
    oracle boundaries is included as a separate upper-bound row.

    Uses LLM-generated per-gloss timestamps from the JSONL.
    Reports overall BLEU + BLEU by position within group.
    """
    output_dir_path = Path(output_dir or cfg.evaluation.results_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Load raw groups (with timestamps) from JSONL ──
    raw_path = cfg.paths.data / "synthetic_discourse.jsonl"
    if not raw_path.exists():
        logger.error(f"Discourse data not found: {raw_path}")
        return {}

    raw_groups = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_groups.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Test split: first test_size groups
    test_raw = raw_groups[:test_size]
    if n_groups and n_groups < len(test_raw):
        test_raw = test_raw[:n_groups]

    # Convert to evaluation format (preserve timestamps)
    groups = []
    for obj in test_raw:
        sents = []
        for s in obj.get("sentences", []):
            if s.get("speaker") == "hearing" and "text" in s:
                entry = {"text": s["text"]}
                if "timestamp_ms" in s:
                    entry["timestamp_ms"] = s["timestamp_ms"]
                sents.append(entry)
            elif "gloss" in s and "text" in s:
                entry = {"gloss": s["gloss"], "text": s["text"]}
                if "gloss_timestamps_ms" in s:
                    entry["gloss_timestamps_ms"] = s["gloss_timestamps_ms"]
                sents.append(entry)
        if len(sents) >= 2 and any("gloss" in s for s in sents):
            groups.append(sents)

    # Check timestamp coverage
    has_ts = sum(1 for g in groups
                 if all("gloss_timestamps_ms" in s for s in g if "gloss" in s))
    logger.info(
        f"Loaded {len(groups)} discourse test groups "
        f"({has_ts} with LLM timestamps, {len(groups) - has_ts} without)"
    )
    if has_ts < len(groups):
        logger.warning(
            "Some groups lack timestamps. Run: "
            "python -m src.training.synthetic_data --mode timestamps"
        )

    # ── Build backend and policies ──
    logger.info(f"Loading backend: {backend_name}, timestamp_mode: {timestamp_mode}")
    backend = build_backend(backend_name)
    policies = build_policies(backend, policy_names)
    logger.info(f"Policies: {list(policies.keys())}")

    # ── Evaluate ──
    all_results = {}
    ts_suffix = f"_{timestamp_mode}" if timestamp_mode != "realistic" else ""
    results_file = output_dir_path / f"eval_{backend_name}_discourse{ts_suffix}.json"
    md_file      = output_dir_path / f"eval_{backend_name}_discourse{ts_suffix}.md"

    # Load existing results for merge
    if results_file.exists():
        try:
            with results_file.open("r") as f:
                all_results = json.load(f).get("policies", {})
            logger.info(f"Loaded {len(all_results)} existing results from {results_file}")
        except Exception:
            pass

    def _save():
        with results_file.open("w") as f:
            json.dump({
                "backend": backend_name,
                "n_groups": len(groups),
                "dataset": "discourse",
                "timestamp_mode": timestamp_mode,
                "policies": all_results,
            }, f, indent=2)
        md_file.write_text(
            _format_discourse_md(all_results, backend_name, len(groups)),
            encoding="utf-8",
        )

    # 1. Batch with oracle boundaries (upper bound)
    if "Batch (oracle)" not in all_results:
        logger.info("Evaluating: Batch (oracle) — upper bound with oracle boundaries...")
        result = await evaluate_policy_on_discourse(
            backend, None, groups,
            context_window=cfg.tlas.context_window_size,
            compute_bertscore=compute_bertscore,
            compute_sbert=compute_sbert,
            compute_chrf=compute_chrf,
            oracle_boundaries=True,
            timestamp_mode=timestamp_mode,
        )
        all_results["Batch (oracle)"] = result
        logger.info(
            f"  Batch (oracle): BLEU={result['bleu']:.2f}  "
            f"ROUGE-L={result['rouge_l']:.4f}  "
            f"n={result['num_sentences']}"
        )
        _save()

    # 2. All policies on continuous stream (realistic evaluation)
    from src.policy.tlas import TLASPolicy
    for pol_name, policy in policies.items():
        if pol_name in all_results:
            logger.info(f"Skipping {pol_name} (already in results)")
            continue

        # Only TLAS variants use discourse context; baselines get none.
        ctx_window = cfg.tlas.context_window_size if isinstance(policy, TLASPolicy) else 0

        logger.info(f"Evaluating: {pol_name} on {len(groups)} groups (continuous stream, ctx={ctx_window})...")
        result = await evaluate_policy_on_discourse(
            backend, policy, groups,
            context_window=ctx_window,
            compute_bertscore=compute_bertscore,
            compute_sbert=compute_sbert,
            compute_chrf=compute_chrf,
            oracle_boundaries=False,
            timestamp_mode=timestamp_mode,
        )
        all_results[pol_name] = result
        logger.info(
            f"  {pol_name}: BLEU={result['bleu']:.2f}  "
            f"ROUGE-L={result['rouge_l']:.4f}  "
            f"n={result['num_sentences']}"
        )
        _save()

    # Print results table
    _print_discourse_results(all_results, backend_name, len(groups))

    await backend.close()
    return all_results


def _format_discourse_md(results: Dict, backend_name: str, n_groups: int) -> str:
    """
    Format discourse evaluation results as a human-readable Markdown file.
    Written alongside the JSON after every policy completes.
    """
    import datetime
    today = datetime.date.today().isoformat()

    # Detect which optional metrics are present
    has_bs    = any(r.get("bertscore_f1") is not None for r in results.values())
    has_sbert = any(r.get("sbert")        is not None for r in results.values())
    has_chrf  = any(r.get("chrf")         is not None for r in results.values())

    n_sentences = next(iter(results.values()), {}).get("num_sentences", "?") if results else "?"

    lines = [
        f"# E2 Discourse Stream Evaluation — {backend_name.upper()} Backend",
        "",
        f"**Date**: {today}",
        f"**Dataset**: `data/synthetic_discourse.jsonl` (test split, first {n_groups} groups)",
        f"**Sentences evaluated**: {n_sentences}",
        "",
        "---",
        "",
        "## Overall Results",
        "",
    ]

    # Build header row
    cols = ["Policy", "BLEU", "ROUGE-L"]
    if has_bs:    cols.append("BERTScore-F1")
    if has_sbert: cols.append("SBERT")
    if has_chrf:  cols.append("chrF++")
    cols.append("Sentences")

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---:" if c != "Policy" else "---" for c in cols]) + "|")

    def _row(name, r):
        cells = [
            name,
            f"{r['bleu']:.2f}",
            f"{r['rouge_l']:.4f}",
        ]
        if has_bs:
            bs = r.get("bertscore_f1")
            cells.append(f"{bs:.4f}" if bs is not None else "N/A")
        if has_sbert:
            sb = r.get("sbert")
            cells.append(f"{sb:.4f}" if sb is not None else "N/A")
        if has_chrf:
            ch = r.get("chrf")
            cells.append(f"{ch:.2f}" if ch is not None else "N/A")
        cells.append(str(r["num_sentences"]))
        return "| " + " | ".join(cells) + " |"

    sep = "|" + "|".join(["---" for _ in cols]) + "|"

    # Canonical order: oracle, then external baselines, then ablations (with separator)
    oracle = results.get("Batch (oracle)")
    if oracle:
        lines.append(_row("Batch (oracle) ↑", oracle))
        lines.append(sep)

    last_was_baseline = True
    for name in POLICY_ORDER:
        if name == "Batch (oracle)" or name not in results:
            continue
        # Insert separator before ablation section
        if last_was_baseline and name in OUR_ABLATIONS:
            lines.append(sep)
            last_was_baseline = False
        lines.append(_row(name, results[name]))

    # Any policy not in POLICY_ORDER (future additions) appended at end
    for name, r in results.items():
        if name not in POLICY_ORDER and name != "Batch (oracle)":
            lines.append(_row(name, r))

    # Position-based BLEU table
    positions = set()
    for r in results.values():
        positions.update(r.get("bleu_by_position", {}).keys())
    positions = sorted(positions)

    if positions:
        lines += ["", "## BLEU by Position within Discourse Group", ""]
        pos_cols = ["Policy"] + [f"Pos {p}" for p in positions]
        lines.append("| " + " | ".join(pos_cols) + " |")
        lines.append("|" + "|".join(["---:" if c != "Policy" else "---" for c in pos_cols]) + "|")

        pos_sep = "|" + "|".join(["---" for _ in pos_cols]) + "|"
        if oracle:
            bp = oracle.get("bleu_by_position", {})
            row = "| Batch (oracle) ↑ | " + " | ".join(f"{bp.get(p, 0.0):.2f}" for p in positions) + " |"
            lines.append(row)
            lines.append(pos_sep)

        last_was_baseline = True
        for name in POLICY_ORDER:
            if name == "Batch (oracle)" or name not in results:
                continue
            if last_was_baseline and name in OUR_ABLATIONS:
                lines.append(pos_sep)
                last_was_baseline = False
            bp = results[name].get("bleu_by_position", {})
            row = f"| {name} | " + " | ".join(f"{bp.get(p, 0.0):.2f}" for p in positions) + " |"
            lines.append(row)

        for name, r in results.items():
            if name not in POLICY_ORDER and name != "Batch (oracle)":
                bp = r.get("bleu_by_position", {})
                row = f"| {name} | " + " | ".join(f"{bp.get(p, 0.0):.2f}" for p in positions) + " |"
                lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _print_discourse_results(results: Dict, backend_name: str, n_groups: int) -> None:
    """Pretty-print discourse evaluation results."""
    # Detect which optional metrics are present
    has_sbert = any(r.get("sbert") is not None for r in results.values())
    has_chrf  = any(r.get("chrf")  is not None for r in results.values())
    has_bs    = any(r.get("bertscore_f1") is not None for r in results.values())

    width = 22 + 8 + 10 + (8 if has_bs else 0) + (8 if has_sbert else 0) + (8 if has_chrf else 0) + 10 + 4
    print(f"\n{'='*width}")
    print(f"Discourse Stream Evaluation — {backend_name} backend, {n_groups} groups")
    print(f"{'='*width}")
    header = f"{'Policy':<22} {'BLEU':>8} {'ROUGE-L':>10}"
    if has_bs:    header += f" {'BERTScore':>9}"
    if has_sbert: header += f" {'SBERT':>8}"
    if has_chrf:  header += f" {'chrF++':>8}"
    header += f" {'Sentences':>10}"
    print(header)
    print(f"{'-'*width}")

    def _row(name, r):
        line = f"{name:<22} {r['bleu']:>8.2f} {r['rouge_l']:>10.4f}"
        if has_bs:
            bs = r.get("bertscore_f1")
            line += f" {bs:>9.4f}" if bs is not None else f" {'N/A':>9}"
        if has_sbert:
            sb = r.get("sbert")
            line += f" {sb:>8.4f}" if sb is not None else f" {'N/A':>8}"
        if has_chrf:
            ch = r.get("chrf")
            line += f" {ch:>8.2f}" if ch is not None else f" {'N/A':>8}"
        line += f" {r['num_sentences']:>10}"
        return line

    # Oracle first, then external baselines, then ablations (with separator)
    oracle = results.get("Batch (oracle)")
    if oracle:
        print(_row("Batch (oracle) ↑", oracle))
        print(f"{'-'*width}")

    last_was_baseline = True
    for name in POLICY_ORDER:
        if name == "Batch (oracle)" or name not in results:
            continue
        if last_was_baseline and name in OUR_ABLATIONS:
            print(f"{'-'*width}  (ablations — this work)")
            last_was_baseline = False
        print(_row(name, results[name]))

    for name, r in results.items():
        if name not in POLICY_ORDER and name != "Batch (oracle)":
            print(_row(name, r))

    # Position-based BLEU
    print(f"\nBLEU by position within discourse group:")
    positions = set()
    for r in results.values():
        positions.update(r.get("bleu_by_position", {}).keys())
    positions = sorted(positions)
    if positions:
        header = f"{'Policy':<22}" + "".join(f" {'Pos '+p:>8}" for p in positions)
        print(header)
        print(f"{'-'*width}")
        if oracle:
            bp = oracle.get("bleu_by_position", {})
            row = f"{'Batch (oracle) ↑':<22}" + "".join(
                f" {bp.get(p, 0.0):>8.2f}" for p in positions)
            print(row)
            print(f"{'-'*width}")
        last_was_baseline = True
        for name in POLICY_ORDER:
            if name == "Batch (oracle)" or name not in results:
                continue
            if last_was_baseline and name in OUR_ABLATIONS:
                print(f"{'-'*width}  (ablations — this work)")
                last_was_baseline = False
            bp = results[name].get("bleu_by_position", {})
            row = f"{name:<22}" + "".join(f" {bp.get(p, 0.0):>8.2f}" for p in positions)
            print(row)
        for name, r in results.items():
            if name not in POLICY_ORDER and name != "Batch (oracle)":
                bp = r.get("bleu_by_position", {})
                row = f"{name:<22}" + "".join(f" {bp.get(p, 0.0):>8.2f}" for p in positions)
                print(row)
    print(f"{'='*width}\n")


# ── Multi-backend comparison ──────────────────────────────────────────────────

async def run_full_comparison(
    backends: Optional[List[str]] = None,
    n_samples: int = 100,
    compute_bertscore: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, MetricResult]]:
    """
    Run evaluation for every specified backend.
    Returns {backend_name: {policy_name: MetricResult}}.
    """
    backends = backends or ["t5", "groq", "gemini", "ollama"]
    all_results = {}
    for bname in backends:
        try:
            all_results[bname] = await run_evaluation(
                backend_name=bname,
                n_samples=n_samples,
                compute_bertscore=compute_bertscore,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for backend {bname}: {e}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate streaming translation policies.")
    p.add_argument("--backend",  type=str, default="t5",
                   choices=["t5", "groq", "gemini", "ollama", "all"],
                   help="Backend to evaluate")
    p.add_argument("--policies", type=str, nargs="+", default=None,
                   help="Policies to evaluate (default: all)")
    p.add_argument("--n",        type=int, default=100,
                   help="Number of test samples")
    p.add_argument("--bertscore", action="store_true", default=False,
                   help="Compute BERTScore (roberta-large token-level F1)")
    p.add_argument("--sbert", action="store_true", default=False,
                   help="Compute Sentence-BERT cosine similarity (all-mpnet-base-v2)")
    p.add_argument("--chrf", action="store_true", default=False,
                   help="Compute chrF++ score (sacrebleu, char+word n-grams)")
    p.add_argument("--output",   type=str, default=None,
                   help="Results output directory")
    p.add_argument("--dataset",  type=str, default="aslg",
                   choices=["aslg", "signum", "discourse"],
                   help="Evaluation dataset (default: aslg)")
    p.add_argument("--test-size", type=int, default=200,
                   help="Discourse test split size (discourse dataset only)")
    p.add_argument("--timestamps", type=str, default="realistic",
                   choices=["realistic", "uniform", "noisy"],
                   help="Timestamp mode for discourse eval: "
                        "realistic (LLM timestamps), "
                        "uniform (constant 450ms, destroys temporal signal), "
                        "noisy (Gaussian jitter, simulates vision latency)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset == "discourse":
        asyncio.run(run_discourse_evaluation(
            backend_name=args.backend,
            policy_names=args.policies,
            n_groups=args.n if args.n != 100 else None,
            test_size=args.test_size,
            compute_bertscore=args.bertscore,
            compute_sbert=args.sbert,
            compute_chrf=args.chrf,
            output_dir=args.output,
            timestamp_mode=args.timestamps,
        ))
    elif args.backend == "all":
        asyncio.run(run_full_comparison(
            n_samples=args.n,
            compute_bertscore=args.bertscore,
            output_dir=args.output,
        ))
    else:
        asyncio.run(run_evaluation(
            backend_name=args.backend,
            policy_names=args.policies,
            n_samples=args.n,
            compute_bertscore=args.bertscore,
            output_dir=args.output,
            dataset=args.dataset,
        ))
