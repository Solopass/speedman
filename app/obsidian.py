r"""Obsidian note exporter and intelligent summarization engine for Speedman.

Formats audio transcripts, speeds, and user-marked 5-second quotes into clean,
structured Obsidian Markdown notes saved directly into the user's primary
vault (D:\OBVLT\1Notebook\Summaries or Digests).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.paths import is_within, safe_stem, to_windows_path

logger = logging.getLogger("speedman_obsidian")

VAULT_NAME = "OBVLT"
VAULT_ROOT = Path("/mnt/d/OBVLT")
NOTEBOOK_ROOT = VAULT_ROOT / "1Notebook"

DEFAULT_FOLDERS = ["Summaries", "Digests", "Research", "Ideas", "Reviews"]
LLM_ROUTER_URL = "http://127.0.0.1:11440/v1"


class ObsidianExportError(RuntimeError):
    pass


def get_available_folders() -> List[str]:
    """Return available subfolders inside 1Notebook, prioritizing standard folders."""
    folders = list(DEFAULT_FOLDERS)
    if NOTEBOOK_ROOT.is_dir():
        try:
            for item in NOTEBOOK_ROOT.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    if item.name not in folders:
                        folders.append(item.name)
        except Exception as e:
            logger.warning(f"Could not read subdirectories from {NOTEBOOK_ROOT}: {e}")
    return folders


def _clean_text_for_summary(segments: List[Dict[str, Any]]) -> str:
    """Extract plain text from segment dictionaries."""
    lines = []
    for s in segments:
        txt = str(s.get("text", "")).strip()
        if txt:
            lines.append(txt)
    return " ".join(lines)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in raw:
        s = s.strip()
        # Keep sentences that have at least 4 words
        if len(s.split()) >= 4:
            cleaned.append(s)
    return cleaned


def _score_sentence(sentence: str, word_freq: Dict[str, int]) -> float:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
    if not words:
        return 0.0
    score = sum(word_freq.get(w, 0) for w in words)
    # Normalize by length with modest penalty for excessive run-ons
    return score / math.sqrt(len(words))


def generate_extractive_summary(text: str, max_sentences: int = 3, max_takeaways: int = 4) -> Tuple[str, List[str]]:
    """Intelligent algorithmic extractive summarizer.

    Used when the local LLM is offline or disabled (e.g. while games / anti-cheat run).
    Ranks sentences by information density and temporal spread across the audio.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        fallback = text.strip()[:300] or "No speech content available for summary."
        return fallback, ["Audio content processed by Speedman."]

    if len(sentences) <= max_sentences:
        return " ".join(sentences), sentences

    # Compute word frequencies (ignoring common stopwords)
    stopwords = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but", "his", "from",
        "they", "say", "her", "she", "will", "one", "all", "would", "there", "their", "what",
        "out", "about", "who", "get", "which", "when", "make", "can", "like", "time", "just",
        "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "are",
        "was", "were", "been", "has", "had", "does", "did", "doing"
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    # Score sentences with position weighting (boost introductory and concluding sentences)
    scored = []
    n = len(sentences)
    for idx, s in enumerate(sentences):
        base_score = _score_sentence(s, freq)
        pos_weight = 1.0
        if idx < max(2, int(n * 0.15)):  # Early context
            pos_weight = 1.3
        elif idx > int(n * 0.85):  # Conclusion / takeaways
            pos_weight = 1.2
        scored.append((idx, base_score * pos_weight, s))

    # Pick top sentences for executive summary
    by_score = sorted(scored, key=lambda x: x[1], reverse=True)
    chosen_summary = sorted(by_score[:max_sentences], key=lambda x: x[0])
    exec_summary = " ".join(item[2] for item in chosen_summary)

    # Pick non-overlapping sentences for key takeaways
    chosen_indices = {item[0] for item in chosen_summary}
    takeaway_cands = [item for item in by_score if item[0] not in chosen_indices]
    takeaway_items = sorted(takeaway_cands[:max_takeaways], key=lambda x: x[0])
    takeaways = [item[2] for item in takeaway_items]

    if not takeaways:
        takeaways = [chosen_summary[0][2]] if chosen_summary else ["Key insights captured from audio."]

    return exec_summary, takeaways


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


def query_local_llm_summary(text: str, timeout_s: float = 25.0) -> Optional[Tuple[str, List[str]]]:
    """Attempt synthesis via SOL local AI router (sol-fast / Gemma 4 on :11440).

    Returns (executive_summary, list_of_takeaways) on success, or None on error/timeout.
    """
    if len(text.strip()) < 50:
        return None

    # Truncate to reasonable context window (~16k characters) if transcript is massive
    truncated_text = text[:18000] if len(text) > 18000 else text

    prompt = (
        "You are an expert audio note summarizer. Analyze the transcript below and provide:\n"
        "1. Executive Summary: 2-4 concise, high-level sentences capturing the main thesis and progression.\n"
        "2. Key Takeaways: 3-5 high-impact bullet points highlighting the most crucial insights.\n\n"
        "Return your response in pure JSON format with exactly two keys: 'summary' (string) and 'takeaways' (array of strings).\n\n"
        f"TRANSCRIPT:\n{truncated_text}"
    )

    req_body = {
        "model": "sol-fast",
        "messages": [
            {"role": "system", "content": "You are a concise, structured research note summarizer. Always respond in valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
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
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed = _parse_json_response(content)
                if parsed:
                    summary = str(parsed.get("summary", "")).strip()
                    takeaways = parsed.get("takeaways", [])
                    if isinstance(takeaways, list) and summary:
                        clean_takeaways = [str(t).strip() for t in takeaways if str(t).strip()]
                        return summary, clean_takeaways
    except Exception as e:
        logger.info(f"Local LLM summarization unavailable or skipped ({e}); utilizing smart extractive summary.")

    return None


def generate_note_summary(
    segments: List[Dict[str, Any]],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Produce executive summary and key takeaways, using local AI if available."""
    text = _clean_text_for_summary(segments)
    if not text:
        return {
            "summary": "No speech content transcribed.",
            "takeaways": ["Audio file contained no recognizable speech."],
            "engine": "none",
        }

    if use_llm:
        llm_res = query_local_llm_summary(text)
        if llm_res:
            summary, takeaways = llm_res
            return {
                "summary": summary,
                "takeaways": takeaways,
                "engine": "sol-fast (Local LLM)",
            }

    exec_summary, takeaways = generate_extractive_summary(text)
    return {
        "summary": exec_summary,
        "takeaways": takeaways,
        "engine": "speedman-extractive (Algorithmic)",
    }


def format_secs(seconds: float | None) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    try:
        sec = max(0.0, float(seconds if seconds is not None else 0.0))
    except (ValueError, TypeError):
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def format_obsidian_markdown(
    title: str,
    source_name: str,
    duration_s: float,
    speed: float,
    summary_data: Optional[Dict[str, Any]] = None,
    bookmarks: Optional[List[Dict[str, Any]]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    custom_tags: Optional[List[str]] = None,
    include_transcript: bool = True,
    transcript_collapsed: bool = True,
    embed_audio: bool = False,
    audio_filename: Optional[str] = None,
    deep_link_timestamps: bool = False,
) -> str:
    """Format comprehensive Obsidian note markdown."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    datetime_str = now.strftime("%Y-%m-%d %H:%M")

    tags = ["speedman", "audio-notes"]
    if custom_tags:
        for t in custom_tags:
            cleaned = re.sub(r"[^\w/-]", "", t.strip().lower())
            if cleaned and cleaned not in tags:
                tags.append(cleaned)

    lines = [
        "---",
        "made_by: speedman",
        f"date: '{date_str}'",
        f"created: '{datetime_str}'",
        "tags:",
    ]
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append(f'source: "{source_name}"')
    if duration_s > 0:
        lines.append(f'duration: "{format_secs(duration_s)}"')
    if speed > 0:
        lines.append(f"playback_speed: {speed:g}x")
    if summary_data and summary_data.get("engine"):
        lines.append(f'summary_engine: "{summary_data["engine"]}"')
    lines.append("---")
    lines.append("")

    # Heading
    lines.append(f"# 🎧 {title}")
    lines.append("")

    # Audio Player Embed
    if embed_audio:
        embed_target = audio_filename or source_name
        lines.append(f"![[{embed_target}]]")
        lines.append("")

    # Summary section
    if summary_data and summary_data.get("summary"):
        lines.append("> [!abstract] Executive Summary")
        for p in summary_data["summary"].split("\n"):
            if p.strip():
                lines.append(f"> {p.strip()}")
        lines.append("")

        takeaways = summary_data.get("takeaways") or []
        if takeaways:
            lines.append("> [!important] Key Takeaways")
            for t in takeaways:
                lines.append(f"> - {t}")
            lines.append("")

    # Helper for deep linking timestamps
    def _format_cite(ts_start: float, ts_end: float) -> str:
        label = f"{format_secs(ts_start)} - {format_secs(ts_end)}"
        if deep_link_timestamps:
            encoded_title = urllib.parse.quote(title)
            return f"[{label}](obsidian://open?vault={VAULT_NAME}&file={encoded_title}#t={int(ts_start)})"
        return f"[{label}]"

    def _format_point(ts: float) -> str:
        label = format_secs(ts)
        if deep_link_timestamps:
            encoded_title = urllib.parse.quote(title)
            return f"[{label}](obsidian://open?vault={VAULT_NAME}&file={encoded_title}#t={int(ts)})"
        return f"[{label}]"

    # Highlights & Quotes section
    bms = bookmarks or []
    if bms:
        lines.append(f"## ⭐ Highlighted Quotes ({len(bms)})")
        lines.append("")
        for bm in bms:
            t_start = bm.get("startTime", max(0.0, bm.get("time", 0.0) - 5.0))
            t_end = bm.get("endTime", bm.get("time", t_start + 5.0))
            quote = str(bm.get("text", "")).strip()
            note = str(bm.get("note", "")).strip()

            lines.append(f'> "{quote}"')
            cite = f"— **{_format_cite(t_start, t_end)}**"
            if note:
                cite += f" *({note})*"
            lines.append(f"> {cite}")
            lines.append("")

    # Full Transcript section
    segs = segments or []
    if include_transcript and segs:
        lines.append("## 📜 Transcript Outline")
        lines.append("")
        if transcript_collapsed:
            lines.append("<details>")
            lines.append("<summary><b>Click to expand full transcript</b></summary>")
            lines.append("")

        for s in segs:
            t = float(s.get("source_start", s.get("start", 0.0)))
            txt = str(s.get("text", "")).strip()
            if txt:
                lines.append(f"- **{_format_point(t)}** {txt}")

        if transcript_collapsed:
            lines.append("")
            lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append(f"*Exported from Speedman on {date_str}*")
    return "\n".join(lines) + "\n"


def append_quotes_to_daily_note(
    quotes: List[Dict[str, Any]],
    source_title: str,
    folder: str = "Digests",
    date_str: Optional[str] = None,
    notebook_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append highlighted quotes to a daily digest note in 1Notebook/<folder>/<YYYY-MM-DD>.md."""
    if not quotes:
        raise ObsidianExportError("No quotes provided to append")

    now = datetime.now()
    day = date_str or now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    base_root = notebook_root if notebook_root else NOTEBOOK_ROOT
    folder_str = str(folder).strip() or "Digests"
    target_dir = (base_root / folder_str).resolve()
    if not is_within(target_dir, base_root):
        raise ObsidianExportError("Target folder is outside permitted 1Notebook directory")

    target_dir.mkdir(parents=True, exist_ok=True)
    note_path = (target_dir / f"{day}.md").resolve()
    if not is_within(note_path, base_root):
        raise ObsidianExportError("Target note path is outside permitted 1Notebook directory")

    lines = []
    # If file doesn't exist, create initial frontmatter
    if not note_path.is_file():
        lines.append("---")
        lines.append("made_by: speedman")
        lines.append(f"date: '{day}'")
        lines.append("tags:")
        lines.append("  - speedman")
        lines.append("  - audio-digests")
        lines.append("---")
        lines.append("")
        lines.append(f"# 📅 Audio Highlights & Daily Digest - {day}")
        lines.append("")

    lines.append(f"### 🎙️ {source_title} ({time_str})")
    lines.append("")
    for bm in quotes:
        t_start = bm.get("startTime", max(0.0, bm.get("time", 0.0) - 5.0))
        t_end = bm.get("endTime", bm.get("time", t_start + 5.0))
        quote = str(bm.get("text", "")).strip()
        note = str(bm.get("note", "")).strip()
        lines.append(f'> "{quote}"')
        cite = f"— **[{format_secs(t_start)} - {format_secs(t_end)}]**"
        if note:
            cite += f" *({note})*"
        lines.append(f"> {cite}")
        lines.append("")

    content_to_append = "\n".join(lines) + "\n"
    if note_path.is_file():
        existing = note_path.read_text(encoding="utf-8")
        note_path.write_text(existing.rstrip() + "\n\n" + content_to_append, encoding="utf-8")
    else:
        note_path.write_text(content_to_append, encoding="utf-8")

    return {
        "status": "success",
        "action": "appended",
        "folder": folder_str,
        "date": day,
        "quotes_count": len(quotes),
        "filename": note_path.name,
        "path": str(note_path),
        "windows_path": to_windows_path(note_path),
    }


def save_obsidian_note(
    note_title: str,
    content: str,
    folder: str = "Summaries",
    notebook_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write note into D:\\OBVLT\\1Notebook\\<folder>\\<note_title>.md safely.

    Validates that the target path resolves inside 1Notebook to prevent path traversal.
    """
    base_root = notebook_root if notebook_root else NOTEBOOK_ROOT
    folder_str = str(folder).strip()

    # Reject path traversal in folder parameter
    if ".." in Path(folder_str).parts or "/" in folder_str or "\\" in folder_str:
        candidate_dir = (base_root / folder_str).resolve()
        if not is_within(candidate_dir, base_root):
            raise ObsidianExportError(f"Target folder '{folder}' is outside the permitted notebook directory.")

    clean_folder = safe_stem(folder_str) or "Summaries"
    target_dir = (base_root / clean_folder).resolve()

    # Security check: must reside inside notebook_root
    if not is_within(target_dir, base_root):
        raise ObsidianExportError(f"Target folder '{folder}' is outside the permitted notebook directory.")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize title for filename
    clean_stem = re.sub(r'[\\/*?:"<>|]', "", note_title).strip()
    if not clean_stem or clean_stem in (".", ".."):
        clean_stem = f"Speedman_Note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    note_path = target_dir / f"{clean_stem}.md"

    # If already exists, write with non-colliding timestamp suffix unless identical content
    if note_path.is_file():
        existing = note_path.read_text(encoding="utf-8")
        if existing != content:
            suffix = datetime.now().strftime("%Y-%m-%d_%H%M")
            note_path = target_dir / f"{clean_stem} ({suffix}).md"

    note_path.write_text(content, encoding="utf-8")

    win_path = to_windows_path(note_path)
    # Generate obsidian:// URI: obsidian://open?vault=OBVLT&file=1Notebook%2FSummaries%2FTitle
    rel_vault_path = f"1Notebook/{clean_folder}/{note_path.stem}"
    encoded_file = urllib.parse.quote(rel_vault_path, safe="")
    obsidian_uri = f"obsidian://open?vault={VAULT_NAME}&file={encoded_file}"

    return {
        "status": "success",
        "title": clean_stem,
        "folder": clean_folder,
        "path": str(note_path),
        "windows_path": win_path,
        "obsidian_uri": obsidian_uri,
        "bytes_written": len(content.encode("utf-8")),
    }
