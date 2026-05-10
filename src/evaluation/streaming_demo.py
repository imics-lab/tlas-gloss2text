"""
Live streaming demo with console timeline visualization.

Shows real-time TLAS decisions as they happen: each gloss token is printed
with its pause/readiness/combined scores, and WRITE decisions are highlighted.

Run:
  python -m src.evaluation.streaming_demo [--backend t5] [--policy tlas]

Or import and call demo_sentence() / demo_file() programmatically.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.config import cfg
from src.pipeline import StreamingPipeline, SentenceResult
from src.policy.afg import PolicyDecision


# ── ANSI colours ──────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")
def _red(t):    return _c(t, "31")
def _dim(t):    return _c(t, "2")


# ── Score bar ─────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


# ── Instrumented pipeline ─────────────────────────────────────────────────────

class DemoPipeline(StreamingPipeline):
    """StreamingPipeline with per-step console output."""

    def __init__(self, backend, policy, speed_multiplier: float = 1.0):
        super().__init__(backend, policy, verbose=False)
        self.speed = speed_multiplier
        self._step_no = 0

    async def _process_sentence(self, gloss_sentence, reference, speed_multiplier):
        """Override to print each step as it happens."""
        self.policy.reset()
        glosses    = gloss_sentence.strip().split()
        n          = len(glosses)
        segments   = []
        t          = 0.0

        import random
        timestamps = []
        for _ in glosses:
            timestamps.append(t)
            dt = max(0.05, random.gauss(
                cfg.evaluation.simulated_intergloss_ms / 1000.0,
                cfg.evaluation.simulated_intergloss_std_ms / 1000.0,
            ))
            t += dt

        print()
        print(_bold(f"  Sentence: {gloss_sentence}"))
        if reference:
            print(_dim(f"  Reference: {reference}"))
        print(_dim("  " + "─" * 70))
        header = (
            f"  {'Gloss':15s} {'P-score':8s} {'R-score':8s} "
            f"{'Combined':9s} {'Decision':10s} {'Reason'}"
        )
        print(_dim(header))
        print(_dim("  " + "─" * 70))

        for i, (gloss, ts) in enumerate(zip(glosses, timestamps)):
            is_final = (i == n - 1)

            if self.speed > 0:
                await asyncio.sleep(
                    (timestamps[i] - (timestamps[i - 1] if i > 0 else 0)) / self.speed
                )

            decision = await self.policy.step(gloss, ts, is_final=is_final)
            self._step_no += 1

            p  = decision.pause_score
            r  = decision.readiness_score
            c  = decision.combined_score
            d  = decision.decision
            is_write = (d == PolicyDecision.WRITE)

            p_bar = _bar(p)
            r_bar = _bar(r)
            c_bar = _bar(c)

            p_str = f"{p:.2f} {p_bar}"
            r_str = f"{r:.2f} {r_bar}"
            c_str = f"{c:.2f}"

            if is_write:
                dec_str = _green(_bold("▶ WRITE"))
            else:
                dec_str = _dim("  READ ")

            reason_short = decision.reason[:30] if decision.reason else ""

            line = (
                f"  {gloss:15s} {p_str:18s} {r_str:18s} "
                f"{c_str:9s} {dec_str:10s} {_dim(reason_short)}"
            )
            if is_write:
                print(_yellow(line))
            else:
                print(line)

            if is_write or is_final:
                buf = list(self.policy.buffer)
                if not buf:
                    buf = [gloss]
                seg = await self._translate_buffer(
                    glosses=buf,
                    timestamps=list(self.policy.timestamps),
                    write_at=ts,
                    write_idx=i,
                    decision=decision,
                )
                if seg and seg.translation:
                    print(_green(f"\n  ▷ [{self.policy.name}]: {seg.translation}\n"))
                    segments.append(seg)
                self.policy.flush()

        print(_dim("  " + "─" * 70))

        from src.pipeline import SentenceResult
        result = SentenceResult(
            reference=reference,
            gloss_input=gloss_sentence,
            segments=segments,
        )
        full = result.full_translation
        print(f"  {_bold('Full translation:')} {_cyan(full)}")
        if reference:
            from src.evaluation.metrics import compute_bleu
            bleu = compute_bleu([full], [reference])
            print(f"  {_dim(f'Sentence BLEU: {bleu:.2f}')}")

        if result.full_translation:
            self._context.append(result.full_translation)
            if len(self._context) > self.context_window:
                self._context.pop(0)
        return result

    async def _step_event(self, event, prior_segments):
        """Override to print each stream event as it arrives."""

        decision = await self.policy.step(
            event.token, event.timestamp, is_final=False
        )
        self._step_no += 1

        p  = decision.pause_score
        r  = decision.readiness_score
        c  = decision.combined_score
        is_write = (decision.decision == PolicyDecision.WRITE)

        p_str = f"{p:.2f} {_bar(p)}"
        r_str = f"{r:.2f} {_bar(r)}"
        dec_str = _green(_bold("▶ WRITE")) if is_write else _dim("  READ ")
        reason_short = decision.reason[:30] if decision.reason else ""
        ts_str = f"[{event.timestamp:7.2f}s]"

        line = (
            f"  {ts_str} {event.token:15s} {p_str:18s} {r_str:18s} "
            f"{c:.2f}  {dec_str}  {_dim(reason_short)}"
        )
        if is_write:
            print(_yellow(line))
        else:
            print(line)

        if is_write:
            buf = list(self.policy.buffer)
            seg = await self._translate_buffer(
                glosses=buf,
                timestamps=list(self.policy.timestamps),
                write_at=event.timestamp,
                write_idx=len(prior_segments),
                decision=decision,
            )
            self.policy.flush()
            if seg and seg.translation:
                print(_green(f"\n  ▷ [{self.policy.name}] +{event.timestamp:.1f}s: {seg.translation}\n"))
            return seg
        return None


# ── Demo runners ──────────────────────────────────────────────────────────────

async def demo_sentence(
    backend,
    policy,
    gloss: str,
    reference: str = "",
    speed_multiplier: float = 2.0,
) -> SentenceResult:
    """Demo translation of a single gloss sentence."""
    pipeline = DemoPipeline(backend, policy, speed_multiplier=speed_multiplier)
    results = await pipeline.run_sentence_list(
        [{"gloss": gloss, "text": reference}],
        speed_multiplier=speed_multiplier,
    )
    return results[0]


async def demo_sample_list(
    backend,
    policy,
    samples: List[Dict],
    speed_multiplier: float = 2.0,
) -> List[SentenceResult]:
    """Demo translation of a list of gloss sentences."""
    pipeline = DemoPipeline(backend, policy, speed_multiplier=speed_multiplier)
    print(_bold(f"\n{'═' * 72}"))
    print(_bold(f"  TLAS Streaming Demo — backend: {backend.name}  policy: {policy.name}"))
    print(_bold(f"{'═' * 72}"))
    return await pipeline.run_sentence_list(samples, speed_multiplier=speed_multiplier)


async def demo_stream_file(
    backend,
    policy,
    stream_path: Union[str, Path],
    speed_multiplier: float = 1.0,
) -> None:
    """Demo translation from a timestamped stream file."""
    pipeline = DemoPipeline(backend, policy, speed_multiplier=speed_multiplier)
    print(_bold(f"\n{'═' * 72}"))
    print(_bold(f"  Live Stream Demo — {stream_path}"))
    print(_bold(f"  Backend: {backend.name}   Policy: {policy.name}"))
    print(_bold(f"{'═' * 72}"))
    header = (
        f"  {'Time':9s} {'Gloss':15s} {'P-score':18s} {'R-score':18s} "
        f"{'Comb':6s} {'Decision':10s} {'Reason'}"
    )
    print(_dim(header))
    print(_dim("  " + "─" * 70))
    segments = await pipeline.run_stream_file(stream_path, speed_multiplier=speed_multiplier)
    print(_dim("  " + "─" * 70))
    print(_bold(f"\n  Total segments: {len(segments)}"))


async def demo_comparison(
    backend,
    samples: List[Dict],
    speed_multiplier: float = 0.0,
) -> None:
    """
    Show all policies side-by-side on the same set of sentences.
    Faster than demo_sample_list — no per-step pause.
    """
    from src.policy.baselines import (
        BatchPolicy, WaitKPolicy, TransLLaMaPolicy, LSGPolicy,
    )
    from src.policy.tlas import TLASPolicy, TLASMode
    from src.evaluation.metrics import compute_bleu, compute_all_metrics
    from src.pipeline import StreamingPipeline

    policies = {
        "Batch":           BatchPolicy(),
        "Wait-3":          WaitKPolicy(k=3),
        "TransLLaMa":      TransLLaMaPolicy(backend),
        "LSG":             LSGPolicy(backend),
        "TLAS":            TLASPolicy(backend, TLASMode.FULL),
        "TLAS-temporal":   TLASPolicy(backend, TLASMode.TEMPORAL_ONLY),
        "TLAS-linguistic": TLASPolicy(backend, TLASMode.LINGUISTIC_ONLY),
    }

    print(_bold(f"\n{'═' * 72}"))
    print(_bold(f"  Policy Comparison — {len(samples)} sentences"))
    print(_bold(f"{'═' * 72}"))

    results = {}
    for name, policy in policies.items():
        pipeline = StreamingPipeline(backend, policy, verbose=False)
        sents = await pipeline.run_sentence_list(samples, speed_multiplier=0.0)
        hyps = [s.full_translation for s in sents]
        refs = [s.reference for s in sents]
        bleu = compute_bleu(hyps, refs)
        results[name] = bleu
        writes = sum(s.num_writes for s in sents) / max(len(sents), 1)
        print(f"  {name:20s}  BLEU={bleu:5.2f}  avg_writes={writes:.2f}")

    print(_bold(f"\n{'─' * 72}"))
    best = max(results, key=results.get)
    print(_bold(f"  Best: {best}  (BLEU={results[best]:.2f})"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live TLAS streaming demo.")
    p.add_argument("--backend", type=str, default="t5",
                   choices=["t5", "groq", "gemini", "ollama"])
    p.add_argument("--policy",  type=str, default="tlas",
                   choices=["batch", "wait_k", "transllama", "lsg", "mma",
                            "tlas", "tlas_temporal", "tlas_linguistic", "compare"])
    p.add_argument("--n",       type=int, default=5,
                   help="Number of test sentences to demo")
    p.add_argument("--speed",   type=float, default=3.0,
                   help="Speed multiplier for simulated timing (0=instant)")
    p.add_argument("--stream",  type=str, default=None,
                   help="Path to a timestamped stream file")
    return p.parse_args()


def _build_backend(name: str):
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
    else:
        from src.backends.ollama_backend import OllamaBackend
        return OllamaBackend()


def _build_policy(name: str, backend):
    from src.policy.baselines import (
        BatchPolicy, WaitKPolicy, TransLLaMaPolicy, LSGPolicy,
    )
    from src.policy.tlas import TLASPolicy, TLASMode
    mapping = {
        "batch":            BatchPolicy(),
        "wait_k":           WaitKPolicy(),
        "transllama":       TransLLaMaPolicy(backend),
        "lsg":              LSGPolicy(backend),
        "tlas":             TLASPolicy(backend, TLASMode.FULL),
        "tlas_temporal":    TLASPolicy(backend, TLASMode.TEMPORAL_ONLY),
        "tlas_linguistic":  TLASPolicy(backend, TLASMode.LINGUISTIC_ONLY),
    }
    return mapping[name.lower()]


async def _main():
    args = _parse_args()
    backend = _build_backend(args.backend)

    if args.stream:
        policy = _build_policy(args.policy, backend)
        await demo_stream_file(backend, policy, args.stream, args.speed)
        await backend.close()
        return

    # Load a few test samples from ASLG-PC12
    from src.data.loader import load_aslg_pc12
    _, _, test = load_aslg_pc12(seed=cfg.training.random_seed)
    samples = test[:args.n]

    if args.policy == "compare":
        await demo_comparison(backend, samples, speed_multiplier=0.0)
    else:
        policy = _build_policy(args.policy, backend)
        await demo_sample_list(backend, policy, samples, speed_multiplier=args.speed)

    await backend.close()


if __name__ == "__main__":
    asyncio.run(_main())
