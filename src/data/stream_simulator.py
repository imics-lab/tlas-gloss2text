"""
Gloss stream simulator.

Parses the timestamped stream files in data/ and replays them as async
generators of StreamEvent objects.  Can also generate synthetic streams
from pre-segmented sentence lists (for offline evaluation).
"""

import asyncio
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np

from src.config import cfg


# ── Data model ────────────────────────────────────────────────────────────────

class SpeakerType(Enum):
    DEAF_GLOSS = "gloss"
    HEARING_TEXT = "text"


@dataclass
class StreamEvent:
    """A single token/gloss arriving in the stream."""
    timestamp: float        # seconds since stream start
    token: str              # gloss word or English word
    speaker: SpeakerType
    raw_line: str = ""


# ── Parser ────────────────────────────────────────────────────────────────────

_LINE_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+"
    r"(Deaf User|Hearing User)\s+\((\w+)\):\s+(.*)"
)


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def parse_stream_file(path: Path) -> List[StreamEvent]:
    """
    Parse a timestamped stream file into a list of StreamEvents.

    Handles both:
      [HH:MM:SS.mmm] Deaf User (Gloss): SINGLE_GLOSS
      [HH:MM:SS.mmm] Hearing User (English): word
    """
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            ts_str, speaker_str, type_str, token = m.groups()
            ts = _parse_timestamp(ts_str)
            speaker = (
                SpeakerType.DEAF_GLOSS
                if "Deaf" in speaker_str
                else SpeakerType.HEARING_TEXT
            )
            # Normalise gloss tokens to uppercase
            if speaker == SpeakerType.DEAF_GLOSS:
                token = token.strip().upper()
            events.append(StreamEvent(ts, token.strip(), speaker, line))

    # Make timestamps relative to first event
    if events:
        t0 = events[0].timestamp
        for e in events:
            e.timestamp -= t0

    return events


# ── Async generators ──────────────────────────────────────────────────────────

async def replay_stream(
    path: Path,
    realtime: bool = True,
    speed: float = 1.0,
    speaker_filter: Optional[SpeakerType] = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Replay a stream file as an async generator.

    Args:
        path:           Path to a *-stream.txt file.
        realtime:       If True, sleep between events to preserve original timing.
        speed:          Playback speed multiplier (>1 = faster). Ignored if not realtime.
        speaker_filter: If set, yield only events from this speaker type.
    """
    events = parse_stream_file(path)
    if not events:
        return

    last_ts = 0.0
    for event in events:
        if speaker_filter and event.speaker != speaker_filter:
            # Still account for time passing even for filtered events
            last_ts = event.timestamp
            continue

        if realtime:
            delay = (event.timestamp - last_ts) / speed
            if delay > 0:
                await asyncio.sleep(delay)

        last_ts = event.timestamp
        yield event


async def synthetic_gloss_stream(
    samples: List[dict],
    intergloss_mean_ms: float = None,
    intergloss_std_ms: float = None,
    inter_sentence_gap_ms: float = 4000.0,
    realtime: bool = False,
    seed: int = 42,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Generate a synthetic gloss stream from a list of {'gloss': ..., 'text': ...} dicts.

    This simulates the upstream vision module for offline evaluation:
    - Within a sentence: inter-gloss gaps drawn from N(mean, std)
    - Between sentences: fixed pause (default 4s, to trigger TPD)

    If realtime=False, timestamps are assigned but no sleeping occurs
    (useful for offline batch evaluation).
    """
    intergloss_mean_ms = intergloss_mean_ms or cfg.evaluation.simulated_intergloss_ms
    intergloss_std_ms  = intergloss_std_ms  or cfg.evaluation.simulated_intergloss_std_ms

    rng = np.random.default_rng(seed)
    current_time = 0.0

    for sample in samples:
        glosses = sample["gloss"].split()
        for i, gloss in enumerate(glosses):
            event = StreamEvent(
                timestamp=current_time,
                token=gloss,
                speaker=SpeakerType.DEAF_GLOSS,
            )
            if realtime and i > 0:
                await asyncio.sleep(intergloss_mean_ms / 1000.0)

            yield event

            # Advance time by a random inter-gloss interval
            gap_ms = rng.normal(intergloss_mean_ms, intergloss_std_ms)
            current_time += max(100.0, gap_ms) / 1000.0

        # Inter-sentence pause (this is the TPD signal)
        current_time += inter_sentence_gap_ms / 1000.0
        if realtime:
            await asyncio.sleep(inter_sentence_gap_ms / 1000.0)


# ── Inter-gloss timing analysis ───────────────────────────────────────────────

def analyse_timing(path: Path, speaker: SpeakerType = SpeakerType.DEAF_GLOSS) -> dict:
    """
    Compute inter-gloss timing statistics from a stream file.
    Useful for calibrating TPD parameters.
    """
    events = [e for e in parse_stream_file(path) if e.speaker == speaker]
    if len(events) < 2:
        return {}

    deltas = np.diff([e.timestamp for e in events])
    return {
        "count": len(events),
        "mean_gap_ms": float(np.mean(deltas) * 1000),
        "std_gap_ms":  float(np.std(deltas) * 1000),
        "median_gap_ms": float(np.median(deltas) * 1000),
        "p90_gap_ms":  float(np.percentile(deltas, 90) * 1000),
        "p99_gap_ms":  float(np.percentile(deltas, 99) * 1000),
        "max_gap_ms":  float(np.max(deltas) * 1000),
    }


if __name__ == "__main__":
    # Quick sanity check
    import json
    for stream_file in [
        cfg.paths.monologue1_stream,
        cfg.paths.monologue2_stream,
        cfg.paths.dialog3_stream,
    ]:
        stats = analyse_timing(stream_file)
        print(f"{stream_file.name}: {json.dumps(stats, indent=2)}")
