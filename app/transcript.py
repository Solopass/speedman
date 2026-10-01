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
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app import timemap_store
from app.paths import safe_stem

logger = logging.getLogger("speedman_transcript")


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


class TranscriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class WarpedTranscript:
    vtt_path: Path
    source_vtt_path: Path
    json_path: Path
    segment_count: int
    source_duration_s: float
    output_duration_s: float
    srt_path: Optional[Path] = None
    source_srt_path: Optional[Path] = None
    txt_path: Optional[Path] = None


def format_timestamp(seconds: float) -> str:
    """WebVTT wants HH:MM:SS.mmm, always with hours and always three decimals."""
    sec = max(0.0, _safe_float(seconds, 0.0))
    total_ms = int(round(sec * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


_TS_RE = re.compile(
    r"(?:(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d+))\s*-->\s*(?:(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d+))"
)


def _parse_ts_match(h, m, s, ms) -> float:
    hours = float(h) if h else 0.0
    frac = float(f"0.{ms}") if ms else 0.0
    return hours * 3600.0 + float(m) * 60.0 + float(s) + frac


def parse_subtitle_content(content: str, filename_hint: str = "") -> list[dict[str, Any]]:
    """Parses WebVTT (.vtt), SubRip (.srt), or JSON subtitle formats into segment dicts.

    Each output segment is guaranteed to have:
      {"start": float, "end": float, "text": str}
    """
    text = content.strip()
    if text.startswith("\ufeff"):
        text = text[1:].strip()

    hint = filename_hint.lower()
    if hint.endswith(".json") or (text.startswith("{") or text.startswith("[")):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                segs = data.get("segments")
                if segs is None:
                    segs = data.get("transcription")
                if segs is None:
                    segs = data.get("words")
                if segs is None:
                    segs = data.get("chunks")
                if segs is None:
                    segs = data.get("results")
                if isinstance(segs, dict):
                    segs = segs.get("segments") or segs.get("words") or []
                if segs is None:
                    segs = []
            elif isinstance(data, list):
                segs = data
            else:
                segs = []
            valid = []
            for s in segs:
                if isinstance(s, dict) and "start" in s and "end" in s:
                    text_val = s.get("text")
                    if text_val is None:
                        text_val = s.get("word")
                    if text_val is None:
                        text_val = s.get("content")
                    if text_val is None:
                        text_val = s.get("transcript", "")
                    clean_text = str(text_val).strip()
                    if clean_text:
                        valid.append({
                            "start": round(float(s["start"]), 3),
                            "end": round(float(s["end"]), 3),
                            "text": clean_text,
                        })
            if not valid and (hint.endswith(".json") or isinstance(data, dict)):
                raise TranscriptError(f"{filename_hint or 'transcript'} has no segments to warp")
            if valid:
                return valid
        except json.JSONDecodeError:
            pass

    blocks = re.split(r"\r?\n\s*\r?\n", text)
    segments: list[dict[str, Any]] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ts_match = None
        ts_line_idx = -1
        for idx, line in enumerate(lines):
            m = _TS_RE.search(line)
            if m:
                ts_match = m
                ts_line_idx = idx
                break
        if not ts_match:
            continue

        start_s = _parse_ts_match(ts_match.group(1), ts_match.group(2), ts_match.group(3), ts_match.group(4))
        end_s = _parse_ts_match(ts_match.group(5), ts_match.group(6), ts_match.group(7), ts_match.group(8))

        sub_lines = lines[ts_line_idx + 1:]
        clean_text = " ".join(sub_lines)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        clean_text = re.sub(r"\{[^\}]+\}", "", clean_text)
        clean_text = clean_text.strip()

        if clean_text and end_s >= start_s:
            segments.append({
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "text": clean_text,
            })

    if not segments:
        raise TranscriptError("Could not extract any timed subtitle segments from input (expected VTT, SRT, or JSON).")

    return segments


def read_segments(transcript_path: Path) -> list[dict[str, Any]]:
    """Reads segments from a file, supporting JSON, WebVTT, and SubRip SRT formats."""
    path = Path(transcript_path)
    if not path.is_file():
        raise TranscriptError(f"transcript not found: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise TranscriptError(f"could not read {path.name}: {e}")
    return parse_subtitle_content(content, filename_hint=path.name)


def estimate_word_times(text: str, start: float, end: float) -> list[dict[str, Any]]:
    """Spread a segment's span across its words, weighted by character count.

    Parakeet gives one timestamp per ~5.4s segment, which at 5x is over a second of
    compressed audio -- a line-level highlight would sit still and then jump a whole
    sentence. Estimating word positions is measured at 0.22s median error against
    YouTube's real per-word timings for the same podcast, or 0.044s once compressed 5x.

    Weighting by length rather than splitting evenly approximates speech: "unfortunately"
    takes longer to say than "a". It is not forced alignment, and it assumes an even
    speaking rate within the segment -- which holds reasonably because parakeet breaks
    segments at pauses, so the long silences fall between segments, not inside them.
    """
    words = str(text).split()
    if not words:
        return []

    span = max(float(end) - float(start), 0.0)
    weights = [max(len(w), 1) for w in words]
    total = float(sum(weights))

    out: list[dict[str, Any]] = []
    cursor = float(start)
    for word, weight in zip(words, weights):
        share = span * weight / total
        out.append({"text": word, "start": cursor, "end": cursor + share})
        cursor += share
    # Absorb float drift into the last word so it ends exactly on the segment boundary.
    out[-1]["end"] = float(end)
    return out


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

        # Words are estimated on the source timeline, then warped through the same map as
        # the segment, so lines and words can never disagree about where they are.
        words = []
        for w in estimate_word_times(text, start, max(end, start)):
            w_start = float(stored.warp(w["start"]))
            w_end = float(stored.warp(w["end"]))
            words.append({
                "text": w["text"],
                "start": round(max(w_start, new_start), 3),
                "end": round(max(w_end, w_start), 3),
                "source_start": round(w["start"], 3),
                "source_end": round(w["end"], 3),
            })

        warped.append({
            "start": round(new_start, 3),
            "end": round(new_end, 3),
            "source_start": round(start, 3),
            "source_end": round(end, 3),
            "text": text,
            "words": words,
        })

    if not warped:
        raise TranscriptError("no usable segments after warping")
    return warped


def format_srt_timestamp(seconds: float) -> str:
    """SubRip (.srt) wants HH:MM:SS,mmm with comma separator."""
    seconds = max(0.0, _safe_float(seconds, 0.0))
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_vtt(warped: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(warped, start=1):
        lines.append(str(i))
        start_s = _safe_float(seg.get("start"), 0.0)
        end_s = _safe_float(seg.get("end"), start_s + 0.05)
        if end_s <= start_s:
            end_s = start_s + 0.05
        lines.append(f"{format_timestamp(start_s)} --> {format_timestamp(end_s)}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines)


def to_source_vtt(warped: list[dict[str, Any]]) -> str:
    """Generate WebVTT on the 1x source timeline (for video player & source audio)."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(warped, start=1):
        lines.append(str(i))
        src_start = _safe_float(seg.get("source_start"), _safe_float(seg.get("start"), 0.0))
        src_end = _safe_float(seg.get("source_end"), _safe_float(seg.get("end"), src_start + 0.05))
        if src_end <= src_start:
            src_end = src_start + 0.05
        lines.append(f"{format_timestamp(src_start)} --> {format_timestamp(src_end)}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines)


def to_srt(segments: list[dict[str, Any]], use_source: bool = False) -> str:
    """Generate SubRip (.srt) formatted subtitle string."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        if use_source:
            start_s = _safe_float(seg.get("source_start"), _safe_float(seg.get("start"), 0.0))
            end_s = _safe_float(seg.get("source_end"), _safe_float(seg.get("end"), start_s + 0.05))
        else:
            start_s = _safe_float(seg.get("start"), 0.0)
            end_s = _safe_float(seg.get("end"), start_s + 0.05)
        if end_s <= start_s:
            end_s = start_s + 0.05
        lines.append(f"{format_srt_timestamp(start_s)} --> {format_srt_timestamp(end_s)}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines)


def to_plain_text(segments: list[dict[str, Any]], include_timestamps: bool = False, use_source: bool = False) -> str:
    """Generate clean plain-text/prose string for note-taking or LLM summarization.

    If include_timestamps is False, clusters continuous speech into flowing paragraphs.
    If True, outputs timestamped lines: [HH:MM:SS] Text...
    """
    if include_timestamps:
        lines = []
        for seg in segments:
            t = _safe_float(seg.get("source_start" if use_source else "start"), 0.0)
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ts = f"[{h:02d}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
            lines.append(f"{ts} {str(seg.get('text', '')).strip()}")
        return "\n".join(lines)

    paragraphs = []
    current_para = []
    prev_end = None

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = _safe_float(seg.get("source_start" if use_source else "start"), 0.0)
        end = _safe_float(seg.get("source_end" if use_source else "end"), start + 0.1)

        if prev_end is not None and (start - prev_end) > 1.2 and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []

        current_para.append(text)
        prev_end = end

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


def sync_segments_to_output(
    segments: list[dict[str, Any]],
    output_name: str,
    output_dir: Optional[Path] = None,
) -> WarpedTranscript:
    """Warp in-memory `segments` onto `output_name`'s timeline and write .vtt + .source.vtt + .synced.json."""
    output_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = safe_stem(output_name)
    if not stem:
        raise TranscriptError(f"Invalid output name '{output_name}'")

    stored = timemap_store.load(output_name)
    warped = warp_segments(segments, stored)

    vtt_path = output_dir / f"{stem}.vtt"
    source_vtt_path = output_dir / f"{stem}.source.vtt"
    srt_path = output_dir / f"{stem}.srt"
    source_srt_path = output_dir / f"{stem}.source.srt"
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.synced.json"

    source_duration = max(s["source_end"] for s in warped) if warped else 0.0
    output_duration = max(s["end"] for s in warped) if warped else 0.0

    vtt_path.write_text(to_vtt(warped), encoding="utf-8")
    source_vtt_path.write_text(to_source_vtt(warped), encoding="utf-8")
    srt_path.write_text(to_srt(warped, use_source=False), encoding="utf-8")
    source_srt_path.write_text(to_srt(warped, use_source=True), encoding="utf-8")
    txt_path.write_text(to_plain_text(warped, include_timestamps=False), encoding="utf-8")

    json_path.write_text(json.dumps({
        "output": str(output_name),
        "speed": stored.speed,
        "sample_rate": stored.sample_rate,
        "warped_with": "uniform_rate" if stored.is_uniform else "time_map",
        "source_duration_s": round(source_duration, 2),
        "output_duration_s": round(output_duration, 2),
        "segments": warped,
    }, indent=2), encoding="utf-8")
    logger.info(f"[transcript] synced {len(warped)} segments to {stem} "
                f"({source_duration:.0f}s -> {output_duration:.0f}s)")

    return WarpedTranscript(
        vtt_path=vtt_path,
        source_vtt_path=source_vtt_path,
        json_path=json_path,
        segment_count=len(warped),
        source_duration_s=round(source_duration, 2),
        output_duration_s=round(output_duration, 2),
        srt_path=srt_path,
        source_srt_path=source_srt_path,
        txt_path=txt_path,
    )


def sync_to_output(
    transcript_json: Path,
    output_name: str,
    output_dir: Optional[Path] = None,
) -> WarpedTranscript:
    """Warp `transcript_json` onto `output_name`'s timeline and write both files."""
    segments = read_segments(transcript_json)
    return sync_segments_to_output(segments, output_name, output_dir)


def _rewrite_transcript_files(
    stem: str,
    data: dict[str, Any],
    output_dir: Path,
) -> None:
    """Helper to rewrite .synced.json, .vtt, .source.vtt, .srt, .source.srt, .txt."""
    segments = data.get("segments", [])
    (output_dir / f"{stem}.synced.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (output_dir / f"{stem}.vtt").write_text(to_vtt(segments), encoding="utf-8")
    (output_dir / f"{stem}.source.vtt").write_text(to_source_vtt(segments), encoding="utf-8")
    (output_dir / f"{stem}.srt").write_text(to_srt(segments, use_source=False), encoding="utf-8")
    (output_dir / f"{stem}.source.srt").write_text(to_srt(segments, use_source=True), encoding="utf-8")
    (output_dir / f"{stem}.txt").write_text(to_plain_text(segments, include_timestamps=False), encoding="utf-8")


def edit_transcript_cue(
    output_name: str,
    line_index: int,
    new_text: str,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Edit text for a single cue in-place, updating word timings and disk files."""
    base_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = safe_stem(output_name)
    if not stem:
        raise TranscriptError(f"Invalid output name '{output_name}'")

    json_path = base_dir / f"{stem}.synced.json"
    if not json_path.is_file():
        raise TranscriptError(f"Synced transcript not found for '{stem}'")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if line_index < 0 or line_index >= len(segments):
        raise TranscriptError(f"Cue index {line_index} out of range (0-{len(segments)-1})")

    seg = segments[line_index]
    seg["text"] = str(new_text).strip()

    start = _safe_float(seg.get("start"), 0.0)
    end = _safe_float(seg.get("end"), start + 0.1)
    src_start = _safe_float(seg.get("source_start"), start)
    src_end = _safe_float(seg.get("source_end"), end)

    raw_words = estimate_word_times(seg["text"], start, end)
    src_words = estimate_word_times(seg["text"], src_start, src_end)
    words = []
    for w_out, w_src in zip(raw_words, src_words):
        words.append({
            "text": w_out["text"],
            "start": round(w_out["start"], 3),
            "end": round(w_out["end"], 3),
            "source_start": round(w_src["start"], 3),
            "source_end": round(w_src["end"], 3),
        })
    seg["words"] = words

    _rewrite_transcript_files(stem, data, base_dir)
    return {
        "status": "success",
        "stem": stem,
        "line_index": line_index,
        "segment": seg,
    }


def batch_find_and_replace(
    output_name: str,
    find_text: str,
    replace_text: str,
    match_case: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Find and replace occurrences across the entire transcript, syncing all files."""
    base_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = safe_stem(output_name)
    if not stem:
        raise TranscriptError(f"Invalid output name '{output_name}'")

    if not find_text:
        raise TranscriptError("find_text cannot be empty")

    json_path = base_dir / f"{stem}.synced.json"
    if not json_path.is_file():
        raise TranscriptError(f"Synced transcript not found for '{stem}'")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    flags = 0 if match_case else re.IGNORECASE
    pattern = re.compile(re.escape(find_text), flags)

    total_replacements = 0
    affected_lines = 0
    modified_cues = []

    for idx, seg in enumerate(segments):
        orig_text = seg.get("text", "")
        new_text, count = pattern.subn(replace_text, orig_text)
        if count > 0:
            total_replacements += count
            affected_lines += 1
            modified_cues.append(idx)
            seg["text"] = new_text

            start = _safe_float(seg.get("start"), 0.0)
            end = _safe_float(seg.get("end"), start + 0.1)
            src_start = _safe_float(seg.get("source_start"), start)
            src_end = _safe_float(seg.get("source_end"), end)
            raw_words = estimate_word_times(new_text, start, end)
            src_words = estimate_word_times(new_text, src_start, src_end)
            words = []
            for w_out, w_src in zip(raw_words, src_words):
                words.append({
                    "text": w_out["text"],
                    "start": round(w_out["start"], 3),
                    "end": round(w_out["end"], 3),
                    "source_start": round(w_src["start"], 3),
                    "source_end": round(w_src["end"], 3),
                })
            seg["words"] = words

    if total_replacements > 0:
        _rewrite_transcript_files(stem, data, base_dir)

    return {
        "status": "success",
        "stem": stem,
        "replacements_count": total_replacements,
        "affected_lines": affected_lines,
        "modified_cues": modified_cues,
        "segments": segments,
    }

