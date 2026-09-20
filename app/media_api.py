"""Media API integration client and local media library scanner for Speedman.

Enables Speedman to seamlessly interact with Media API (127.0.0.1:8080):
- Downloading/extracting audio from YouTube and web URLs
- Browsing media files previously ingested by Media API
- Sending sped-up audio back to Media API for Speech-To-Text transcription
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

from app.paths import to_windows_path

logger = logging.getLogger("speedman_media_api")

_HTTP_URL_RE = re.compile(r"^https?://\S", re.IGNORECASE)

MEDIA_API_BASE = os.getenv("MEDIA_API_URL", "http://127.0.0.1:8080")
MEDIA_API_AUDIO_DIR = Path("/mnt/d/Output/Audio")
MEDIA_API_VIDEO_DIR = Path("/mnt/d/Output/Videos")
DOWNLOADS_DIR = Path("/mnt/d/Audio/Speed/downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


_ONLINE_CACHE_TTL = 10.0
_online_cache: tuple[float, bool] = (0.0, False)


def check_media_api_online(force: bool = False) -> bool:
    """Returns True if Media API on port 8080 is reachable.

    Cached for _ONLINE_CACHE_TTL seconds: /health calls this, and an uncached miss costs
    the full 1.5s timeout whenever Media API is down. Pass force=True to re-probe.
    """
    global _online_cache
    checked_at, cached = _online_cache
    now = time.monotonic()
    if not force and now - checked_at < _ONLINE_CACHE_TTL:
        return cached

    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{MEDIA_API_BASE}/api/v1/health")
            online = resp.status_code == 200
    except Exception:
        online = False

    _online_cache = (now, online)
    return online


def get_media_api_health() -> Optional[Dict[str, Any]]:
    """Returns Media API health status dict or None if offline."""
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{MEDIA_API_BASE}/api/v1/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def list_media_library(limit: int = 30) -> List[Dict[str, Any]]:
    """Scans Media API output directories and Speedman downloads for available audio."""
    candidates: List[Path] = []
    search_dirs = [MEDIA_API_AUDIO_DIR, DOWNLOADS_DIR, MEDIA_API_VIDEO_DIR]

    for d in search_dirs:
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".webm"):
                    candidates.append(p)

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:  # file vanished between iterdir() and stat()
            return 0.0

    candidates.sort(key=mtime, reverse=True)

    items = []
    for p in candidates[:limit]:
        try:
            st = p.stat()
        except OSError:
            continue
        items.append({
            "filename": p.name,
            "path": str(p),
            "windows_path": to_windows_path(p),
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "modified": time.ctime(st.st_mtime),
            "category": "audio" if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus") else "video",
        })
    return items


def extract_audio_from_url(url: str) -> Path:
    """Extracts audio from a YouTube or web media URL by invoking yt-dlp on E-cores
    into DOWNLOADS_DIR.

    (Delegating to Media API when it is up would save a second download, but that path
    is not implemented -- this always shells out to yt-dlp.)
    """
    url = str(url or "").strip()
    # yt-dlp takes the URL as a positional argument, so a value beginning with "-" would
    # be parsed as an option (--exec, --config-location, ...). The scheme check plus the
    # "--" terminator below keep caller input out of yt-dlp's option parser.
    if not _HTTP_URL_RE.match(url):
        raise ValueError("Only http:// and https:// URLs can be ingested")

    logger.info(f"[media_api] Extracting audio for URL: {url}")

    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        # Check standard paths
        for candidate in ["/home/jake/.local/bin/yt-dlp", "/usr/local/bin/yt-dlp"]:
            if Path(candidate).is_file():
                ytdlp_bin = candidate
                break

    if not ytdlp_bin:
        raise RuntimeError("yt-dlp is not installed or available on PATH")

    out_template = str(DOWNLOADS_DIR / "%(title).200B.%(ext)s")

    # E-core pinning is a nicety, not a requirement: if the tools are missing, download
    # unpinned rather than failing with FileNotFoundError on taskset.
    prefix: List[str] = []
    if shutil.which("taskset"):
        prefix += ["taskset", "-c", "8-15"]
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]

    cmd = [
        *prefix,
        ytdlp_bin,
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
        "--",  # end of options; everything after this is a positional URL
        url,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"yt-dlp download failed: {err}")

    # Find the printed path
    lines = [ln.strip() for ln in proc.stdout.strip().split("\n") if ln.strip()]
    if lines:
        downloaded = Path(lines[-1])
        if downloaded.is_file():
            return downloaded

    # Fallback: scan DOWNLOADS_DIR for newest WAV
    wavs = sorted(DOWNLOADS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        return wavs[0]

    raise RuntimeError("Audio was downloaded but target file could not be determined")


def send_to_media_api_transcribe(audio_path: Path, engine: str = "parakeet") -> Dict[str, Any]:
    """Posts an audio file to Media API for Speech-To-Text transcription."""
    if not check_media_api_online():
        raise RuntimeError("Media API (127.0.0.1:8080) is currently offline")

    payload = {
        "source": str(audio_path),
        "engine": engine,
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{MEDIA_API_BASE}/api/v1/transcribe", json=payload)
        if resp.status_code not in (200, 201, 202):
            err = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Media API transcription request failed: {err}")
        return resp.json()
