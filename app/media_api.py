"""Media API integration client and local media library scanner for Speedman.

Enables Speedman to seamlessly interact with Media API (127.0.0.1:8080):
- Downloading/extracting audio from YouTube and web URLs
- Browsing media files previously ingested by Media API
- Sending sped-up audio back to Media API for Speech-To-Text transcription
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger("speedman_media_api")

MEDIA_API_BASE = os.getenv("MEDIA_API_URL", "http://127.0.0.1:8080")
MEDIA_API_AUDIO_DIR = Path("/mnt/d/Output/Audio")
MEDIA_API_VIDEO_DIR = Path("/mnt/d/Output/Videos")
DOWNLOADS_DIR = Path("/mnt/d/Audio/Speed/downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def check_media_api_online() -> bool:
    """Returns True if Media API on port 8080 is reachable."""
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{MEDIA_API_BASE}/api/v1/health")
            return resp.status_code == 200
    except Exception:
        return False


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

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    items = []
    for p in candidates[:limit]:
        st = p.stat()
        items.append({
            "filename": p.name,
            "path": str(p),
            "windows_path": f"D:\\{p.relative_to(Path('/mnt/d')).as_posix().replace('/', '\\')}",
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "modified": time.ctime(st.st_mtime),
            "category": "audio" if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus") else "video",
        })
    return items


def extract_audio_from_url(url: str) -> Path:
    """Extracts audio from a YouTube or web media URL.
    
    If Media API is running, delegates to Media API.
    Otherwise, invokes yt-dlp directly on E-cores into DOWNLOADS_DIR.
    """
    logger.info(f"[media_api] Extracting audio for URL: {url}")

    # Fallback to local yt-dlp on E-cores
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
    cmd = [
        "taskset", "-c", "8-15",
        "nice", "-n", "10",
        ytdlp_bin,
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
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
