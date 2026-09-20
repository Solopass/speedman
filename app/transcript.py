"""Warp a 1x transcript onto the compressed timeline, and write it as WebVTT + JSON.

Transcription runs on the source, because ASR collapses on time-compressed speech (150
coherent words at 1x against 11 words of nonsense at 5x -- docs/EVALUATION.md). That
leaves an accurate transcript whose timestamps refer to the original. Pushing them
through the stored time map produces one that stays synced while you listen at 5x.

WebVTT because a .vtt is understood natively by the <video>/<audio> element, by VLC, and
by most players, so the result is useful outside Speedman with no extra code. The JSON
keeps sample-accurate positions for anything programmatic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app import timemap_store

logger = logging.getLogger("speedman_transcript")


class TranscriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class WarpedTranscript:
    vtt_path: Path
    json_path: Path
    segment_count: int
    source_duration_s: float
    output_duration_s: float


def format_timestamp(seconds: float) -> str:
    """WebVTT wants HH:MM:SS.mmm, always with hours and always three decimals."""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def read_segments(transcript_json: Path) -> list[dict[str, Any]]:
    """Media API's shape: {"segments": [{"start", "end", "text"}, ...]}."""
    path = Path(transcript_json)
    if not path.is_file():
        raise TranscriptError(f"transcript not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TranscriptError(f"could not parse {path.name}: {e}")

    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise TranscriptError(f"{path.name} has no segments to warp")
    return segments


def warp_segments(segments: list[dict[str, Any]], stored: timemap_store.StoredTimeMap):
    """Map each segment's start and end onto the compressed timeline."""
    warped = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        new_start = float(stored.warp(start))
        new_end = float(stored.warp(max(end, start)))
        # A cue must have positive duration or players drop it. Segments that fall
        # entirely inside a hard-compressed pause can collapse to the same instant.
        if new_end <= new_start:
            new_end = new_start + 0.05

        warped.append({
            "start": round(new_start, 3),
            "end": round(new_end, 3),
            "source_start": round(start, 3),
            "source_end": round(end, 3),
            "text": text,
        })

    if not warped:
        raise TranscriptError("no usable segments after warping")
    return warped


def to_vtt(warped: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(warped, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def sync_to_output(
    transcript_json: Path,
    output_name: str,
    output_dir: Optional[Path] = None,
) -> WarpedTranscript:
    """Warp `transcript_json` onto `output_name`'s timeline and write both files.

    Written beside the compressed audio and named after it, so a player asked to load
    subtitles for `talk_5x_fast.mp3` finds `talk_5x_fast.vtt` without being told.
    """
    output_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = Path(str(output_name)).stem

    stored = timemap_store.load(output_name)
    segments = read_segments(transcript_json)
    warped = warp_segments(segments, stored)

    vtt_path = output_dir / f"{stem}.vtt"
    json_path = output_dir / f"{stem}.synced.json"

    vtt_path.write_text(to_vtt(warped), encoding="utf-8")
    json_path.write_text(json.dumps({
        "output": str(output_name),
        "speed": stored.speed,
        "sample_rate": stored.sample_rate,
        "warped_with": "uniform_rate" if stored.is_uniform else "time_map",
        "segments": warped,
    }, indent=2), encoding="utf-8")

    source_duration = max(s["source_end"] for s in warped)
    output_duration = max(s["end"] for s in warped)
    logger.info(f"[transcript] synced {len(warped)} segments to {stem} "
                f"({source_duration:.0f}s -> {output_duration:.0f}s)")

    return WarpedTranscript(
        vtt_path=vtt_path,
        json_path=json_path,
        segment_count=len(warped),
        source_duration_s=round(source_duration, 2),
        output_duration_s=round(output_duration, 2),
    )
