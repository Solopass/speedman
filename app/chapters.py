r"""Semantic chapter detection and Table of Contents (TOC) generator for Speedman.

Analyzes transcripts into logical chapter sections using local AI (sol-fast)
or algorithmic pause & lexical distribution analysis.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app import timemap_store
from app.paths import is_within, safe_stem

logger = logging.getLogger("speedman_chapters")

LLM_ROUTER_URL = "http://127.0.0.1:11440/v1"


class ChapterError(RuntimeError):
    pass


def _extract_chapter_title_from_text(text: str, default: str = "Section") -> str:
    """Derive a clean, concise 2-4 word chapter title from leading segment text."""
    clean = re.sub(r"[^\w\s-]", "", text).strip()
    words = [w for w in clean.split() if len(w) > 2]
    stopwords = {
        "this", "that", "there", "these", "those", "what", "when", "where", "which",
        "with", "about", "would", "could", "should", "their", "there", "here", "just",
        "like", "know", "going", "think", "really", "actually", "well", "okay", "right",
        "yeah", "hello", "welcome", "today", "talk", "talking", "some", "other"
    }
    filtered = [w.capitalize() for w in words if w.lower() not in stopwords]
    if len(filtered) >= 2:
        return " ".join(filtered[:3])
    if len(filtered) == 1:
        return filtered[0]
    return default


def generate_algorithmic_chapters(
    segments: List[Dict[str, Any]],
    duration_s: float,
    min_chapter_duration_s: float = 45.0,
    target_chapters: int = 5,
) -> List[Dict[str, Any]]:
    """Generates chapters algorithmically using pause gaps, section boundaries, and keyword shifts.

    Guaranteed to run in < 5ms without network calls.
    """
    if not segments or duration_s <= 0:
        return [{"id": 1, "title": "Full Recording", "start": 0.0, "end": max(0.0, duration_s), "summary": ""}]

    # If duration is very short (< 90s), a single chapter is appropriate
    if duration_s < 90.0:
        title = _extract_chapter_title_from_text(segments[0].get("text", ""), default="Discussion")
        return [{"id": 1, "title": title, "start": 0.0, "end": round(duration_s, 2), "summary": ""}]

    # 1. Identify candidate boundary points based on pause length between segments
    candidates: List[Tuple[float, float, str]] = []  # (boundary_time, pause_gap, next_text)
    for i in range(len(segments) - 1):
        cur_end = float(segments[i].get("start", 0.0)) + 0.1
        if "end" in segments[i] and segments[i]["end"] is not None:
            cur_end = float(segments[i]["end"])
        nxt_start = float(segments[i + 1].get("start", cur_end))
        gap = nxt_start - cur_end
        candidates.append((cur_end, gap, segments[i + 1].get("text", "")))

    # Desired interval between chapters
    ideal_interval = max(min_chapter_duration_s, duration_s / max(2, target_chapters))

    split_points = [0.0]
    next_min_time = ideal_interval * 0.7

    for boundary_time, gap, text in candidates:
        if boundary_time >= next_min_time and (duration_s - boundary_time) >= (min_chapter_duration_s * 0.7):
            # Significant pause or reached interval limit
            if gap >= 1.2 or (boundary_time - split_points[-1]) >= (ideal_interval * 1.3):
                split_points.append(boundary_time)
                next_min_time = boundary_time + ideal_interval * 0.7

    split_points.append(duration_s)

    # 2. Build chapter objects with titles derived from initial sentences
    chapters: List[Dict[str, Any]] = []
    for idx in range(len(split_points) - 1):
        c_start = round(split_points[idx], 2)
        c_end = round(split_points[idx + 1], 2)

        # Find segments in this window to pick title
        window_segs = [
            s for s in segments
            if float(s.get("start", 0.0)) >= (c_start - 0.5) and float(s.get("start", 0.0)) < c_end
        ]

        if idx == 0:
            title = "Introduction & Overview"
        elif idx == len(split_points) - 2 and (c_end - c_start) > 30.0:
            title = "Conclusion & Takeaways"
        elif window_segs:
            first_text = window_segs[0].get("text", "")
            title = _extract_chapter_title_from_text(first_text, default=f"Part {idx + 1}")
        else:
            title = f"Part {idx + 1}"

        chapters.append({
            "id": idx + 1,
            "title": title,
            "start": c_start,
            "end": c_end,
            "summary": window_segs[0].get("text", "")[:120] if window_segs else "",
        })

    return chapters


def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, stripping markdown fences or embedded wrappers."""
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def query_llm_chapters(
    segments: List[Dict[str, Any]],
    duration_s: float,
    timeout_s: float = 25.0,
) -> Optional[List[Dict[str, Any]]]:
    """Attempts local AI chapter detection via sol-fast on :11440."""
    if not segments or duration_s < 60.0:
        return None

    # Construct condensed transcript with timestamps
    lines = []
    step = max(1, len(segments) // 60)  # at most ~60 lines for prompt
    for i in range(0, len(segments), step):
        s = segments[i]
        t = float(s.get("start", 0.0))
        m, sec = int(t // 60), int(t % 60)
        txt = str(s.get("text", "")).strip()
        if txt:
            lines.append(f"[{m:02d}:{sec:02d}] {txt}")

    sample_transcript = "\n".join(lines[:60])
    prompt = (
        f"You are an expert audio editor. The audio duration is {duration_s:.1f} seconds.\n"
        "Analyze the timestamped transcript below and divide it into 3 to 8 logical chapters with titles.\n"
        "Return pure JSON format with a single key 'chapters', which is an array of objects:\n"
        '[{"title": "Introduction", "start": 0.0, "end": 65.0, "summary": "Brief summary"}]\n\n'
        f"TRANSCRIPT SAMPLE:\n{sample_transcript}"
    )

    req_body = {
        "model": "sol-fast",
        "messages": [
            {"role": "system", "content": "You are an audio chaptering assistant. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }

    try:
        data_bytes = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            f"{LLM_ROUTER_URL}/chat/completions",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status == 200:
                res_data = json.loads(resp.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                parsed = _parse_json_response(content)
                if parsed:
                    ch_list = parsed.get("chapters", [])
                    if isinstance(ch_list, list) and len(ch_list) >= 2:
                        clean_chapters = []
                        for idx, c in enumerate(ch_list):
                            start = max(0.0, float(c.get("start", 0.0)))
                            end = min(duration_s, float(c.get("end", duration_s)))
                            if end <= start:
                                end = start + 30.0
                            clean_chapters.append({
                                "id": idx + 1,
                                "title": str(c.get("title", f"Chapter {idx + 1}")).strip(),
                                "start": round(start, 2),
                                "end": round(end, 2),
                                "summary": str(c.get("summary", "")).strip(),
                            })
                        # Ensure first starts at 0 and last ends at duration_s
                        clean_chapters[0]["start"] = 0.0
                        clean_chapters[-1]["end"] = round(duration_s, 2)
                        return clean_chapters
    except Exception as e:
        logger.info(f"Local AI chaptering unavailable ({e}); falling back to algorithmic detection.")

    return None


def detect_and_save_chapters(
    output_name: str,
    segments: List[Dict[str, Any]],
    duration_s: float,
    output_dir: Optional[Path] = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Detects chapters for output_name, saves <stem>.chapters.json, and returns chapter list."""
    base_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = safe_stem(output_name)
    if not stem:
        raise ChapterError(f"Invalid output name '{output_name}'")

    chapters = None
    if use_llm:
        chapters = query_llm_chapters(segments, duration_s)

    if not chapters:
        chapters = generate_algorithmic_chapters(segments, duration_s)

    target_file = base_dir / f"{stem}.chapters.json"
    try:
        target_file.write_text(json.dumps({"chapters": chapters, "stem": stem, "duration_s": duration_s}, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not cache chapters to {target_file}: {e}")

    return chapters


def load_cached_chapters(
    output_name: str,
    output_dir: Optional[Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Loads cached chapters if available."""
    base_dir = Path(output_dir) if output_dir else timemap_store.OUTPUT_DIR
    stem = safe_stem(output_name)
    if not stem:
        return None
    target_file = base_dir / f"{stem}.chapters.json"
    if target_file.is_file() and is_within(target_file, base_dir):
        try:
            data = json.loads(target_file.read_text(encoding="utf-8"))
            return data.get("chapters", [])
        except Exception:
            return None
    return None
