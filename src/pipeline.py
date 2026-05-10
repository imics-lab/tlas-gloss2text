"""
Main async streaming pipeline for TLAS.

Orchestrates:
  - Stream ingestion (real stream file or synthetic simulation)
  - Policy decision (any policy conforming to step()/flush()/reset())
  - Backend translation (any TranslationBackend)
  - Discourse context window management
  - Metrics collection for offline evaluation

Usage (programmatic):
    pipeline = StreamingPipeline(backend, policy)
    results  = await pipeline.run_sentence_list(test_samples)

Usage (live stream file):
    pipeline = StreamingPipeline(backend, policy)
    results  = await pipeline.run_stream_file(path_to_stream_file)
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.backends.base import Direction, TranslationResult
from src.config import cfg
from src.data.stream_simulator import StreamEvent, SpeakerType
from src.policy.afg import PolicyDecision

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """A single translated segment (one WRITE decision)."""
    glosses:         List[str]          # source glosses
    timestamps:      List[float]        # per-gloss timestamps
    translation:     str                # backend output
    reference:       Optional[str]      # ground truth if available
    write_at:        float              # wall-clock time of WRITE decision
    write_gloss_idx: int                # index of the triggering gloss
    reason:          str                # AFG reason string
    pause_score:     float
    readiness_score: float
    combined_score:  float


@dataclass
class SentenceResult:
    """All segments produced for one ground-truth sentence."""
    reference:   str
    gloss_input: str
    segments:    List[SegmentResult] = field(default_factory=list)

    @property
    def full_translation(self) -> str:
        """Concatenate all segments into one string."""
        return " ".join(s.translation for s in self.segments if s.translation.strip())

    @property
    def num_writes(self) -> int:
        return len(self.segments)

    @property
    def average_lagging(self) -> float:
        """
        Average Lagging (AL) per Ma et al. (2019).
        AL = (1/|Y|) Σ_t  g(t) - (t-1)
        where g(t) is the source position read before writing output token t.
        We approximate using segment-level read positions.
        """
        if not self.segments:
            return 0.0
        # Compute total output tokens across segments
        total_out_tokens = sum(len(s.translation.split()) for s in self.segments)
        if total_out_tokens == 0:
            return 0.0
        # Compute cumulative read positions at each write
        al_sum = 0.0
        cumulative_read = 0
        cumulative_out  = 0
        for seg in self.segments:
            cumulative_read += len(seg.glosses)
            seg_out = len(seg.translation.split())
            for t in range(1, seg_out + 1):
                al_sum += cumulative_read - (cumulative_out + t - 1)
            cumulative_out += seg_out
        return al_sum / total_out_tokens


# ── Pipeline ──────────────────────────────────────────────────────────────────

class StreamingPipeline:
    """
    Runs any (backend, policy) pair over a stream of glosses.

    Thread-safety: not thread-safe; create one instance per concurrent stream.
    """

    def __init__(
        self,
        backend,
        policy,
        direction: Direction = Direction.GLOSS_TO_TEXT,
        context_window: int = None,
        verbose: bool = False,
    ):
        self.backend        = backend
        self.policy         = policy
        self.direction      = direction
        self.context_window = cfg.tlas.context_window_size if context_window is None else context_window
        self.verbose        = verbose

        # Discourse context: ring buffer of recent translations
        self._context: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_sentence_list(
        self,
        samples: List[Dict],
        speed_multiplier: float = 0.0,
        progress: bool = True,
    ) -> List[SentenceResult]:
        """
        Evaluate over a list of {"gloss": ..., "text": ...} samples.

        Args:
            samples:          List of dicts with "gloss" and "text" keys.
            speed_multiplier: 0 = no sleep (fast eval); >0 = simulate real-time pacing.
            progress:         Show a tqdm progress bar (default True).
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc=self.policy.name, unit="sent", leave=False)
        except ImportError:
            iterator = samples

        self._context.clear()
        results = []
        for sample in iterator:
            result = await self._process_sentence(
                gloss_sentence=sample["gloss"],
                reference=sample.get("text", ""),
                speed_multiplier=speed_multiplier,
            )
            results.append(result)
            # Update discourse context
            if result.full_translation:
                self._context.append(result.full_translation)
                if len(self._context) > self.context_window:
                    self._context.pop(0)
        return results

    async def run_stream_file(
        self,
        stream_path: Union[str, Path],
        speed_multiplier: float = 1.0,
    ) -> List[SegmentResult]:
        """
        Process a timestamped stream file.  Returns flat list of segments
        (sentences are determined dynamically by the policy).
        """
        from src.data.stream_simulator import replay_stream

        self._context.clear()
        self.policy.reset()
        all_segments: List[SegmentResult] = []

        async for event in replay_stream(Path(stream_path), speed=speed_multiplier):
            if event.speaker == SpeakerType.HEARING_TEXT:
                # Skip hearing-side tokens in gloss→text mode
                continue
            segment = await self._step_event(event, all_segments)
            if segment:
                all_segments.append(segment)

        # Flush any remaining buffer
        final_seg = await self._force_flush(timestamp=asyncio.get_event_loop().time())
        if final_seg:
            all_segments.append(final_seg)

        return all_segments

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _process_sentence(
        self,
        gloss_sentence: str,
        reference: str,
        speed_multiplier: float,
    ) -> SentenceResult:
        """Stream one sentence through the policy and collect results."""
        self.policy.reset()
        glosses = gloss_sentence.strip().split()
        n = len(glosses)
        segments: List[SegmentResult] = []

        # Assign synthetic timestamps
        t = 0.0
        timestamps = []
        for _ in glosses:
            timestamps.append(t)
            dt = max(0.05, random.gauss(
                cfg.evaluation.simulated_intergloss_ms / 1000.0,
                cfg.evaluation.simulated_intergloss_std_ms / 1000.0,
            ))
            t += dt
            if speed_multiplier > 0:
                await asyncio.sleep(dt / speed_multiplier)

        for i, (gloss, ts) in enumerate(zip(glosses, timestamps)):
            is_final = (i == n - 1)
            decision = await self.policy.step(gloss, ts, is_final=is_final)

            if decision.decision == PolicyDecision.WRITE or is_final:
                # Collect accumulated buffer from policy
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
                if seg:
                    segments.append(seg)
                self.policy.flush()

        result = SentenceResult(
            reference=reference,
            gloss_input=gloss_sentence,
            segments=segments,
        )
        if self.verbose:
            logger.info(
                f"[{self.policy.name}] {repr(gloss_sentence)[:40]} → "
                f"{repr(result.full_translation)[:60]}  ({len(segments)} writes)"
            )
        return result

    async def _step_event(
        self,
        event: StreamEvent,
        prior_segments: List[SegmentResult],
    ) -> Optional[SegmentResult]:
        """Process one stream event; return a SegmentResult if a WRITE occurred."""
        decision = await self.policy.step(
            event.token, event.timestamp, is_final=False
        )
        if decision.decision == PolicyDecision.WRITE:
            buf = list(self.policy.buffer)
            seg = await self._translate_buffer(
                glosses=buf,
                timestamps=list(self.policy.timestamps),
                write_at=event.timestamp,
                write_idx=len(prior_segments),
                decision=decision,
            )
            self.policy.flush()
            return seg
        return None

    async def _force_flush(self, timestamp: float) -> Optional[SegmentResult]:
        """Translate remaining buffer (end-of-stream)."""
        buf = list(self.policy.buffer)
        if not buf:
            return None
        from src.policy.afg import AFGDecision, PolicyDecision as PD
        dummy = AFGDecision(PD.WRITE, 0.0, 0.0, 0.0, "end-of-stream flush")
        return await self._translate_buffer(
            glosses=buf,
            timestamps=list(self.policy.timestamps),
            write_at=timestamp,
            write_idx=-1,
            decision=dummy,
        )

    async def _translate_buffer(
        self,
        glosses: List[str],
        timestamps: List[float],
        write_at: float,
        write_idx: int,
        decision,
    ) -> Optional[SegmentResult]:
        """Call backend.translate() with optional discourse context."""
        if not glosses:
            return None
        gloss_str   = " ".join(glosses)
        context_str = " ||| ".join(self._context[-self.context_window:]) if self._context else ""

        try:
            result: TranslationResult = await self.backend.translate(
                gloss_str,
                context=context_str,
                direction=self.direction,
            )
            translation = result.translation.strip()
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            translation = ""

        return SegmentResult(
            glosses=glosses,
            timestamps=timestamps,
            translation=translation,
            reference=None,
            write_at=write_at,
            write_gloss_idx=write_idx,
            reason=decision.reason,
            pause_score=decision.pause_score,
            readiness_score=decision.readiness_score,
            combined_score=decision.combined_score,
        )


# ── Convenience function ──────────────────────────────────────────────────────

async def quick_translate(
    backend,
    gloss: str,
    policy=None,
) -> str:
    """
    Translate a single gloss sentence using BatchPolicy (no streaming).
    Useful for sanity-checks and qualitative demos.
    """
    from src.policy.baselines import BatchPolicy
    policy = policy or BatchPolicy()
    pipeline = StreamingPipeline(backend, policy)
    results = await pipeline.run_sentence_list([{"gloss": gloss, "text": ""}])
    return results[0].full_translation if results else ""
