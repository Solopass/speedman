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

_UPDATE_CHECK_INTERVAL_S = float(os.getenv("SPEEDMAN_YTDLP_UPDATE_INTERVAL", "86400"))  # 24 hours
_LAST_UPDATE_CHECK_FILE = Path("/mnt/d/Audio/Speed/.ytdlp_last_update_check")


def should_check_ytdlp_update() -> bool:
    """True if yt-dlp has not been checked for updates within the last 24 hours."""
    if not _LAST_UPDATE_CHECK_FILE.exists():
        return True
    try:
        last_ts = float(_LAST_UPDATE_CHECK_FILE.read_text(encoding="utf-8").strip())
        return (time.time() - last_ts) >= _UPDATE_CHECK_INTERVAL_S
    except Exception:
        return True


def record_ytdlp_update_checked() -> None:
    """Persists timestamp of the last yt-dlp update check."""
    try:
        _LAST_UPDATE_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_UPDATE_CHECK_FILE.write_text(str(time.time()), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[media_api] Failed to record yt-dlp update check timestamp: {e}")


def ensure_ytdlp_updated(force: bool = False) -> Dict[str, Any]:
    """Checks and updates yt-dlp at most once a day when called.

    If Media API is online, delegates to its auto-updating endpoint.
    If Media API is offline or fails, falls back to updating the local yt-dlp binary.
    """
    if not force and not should_check_ytdlp_update():
        return {"status": "skipped", "message": "Checked recently (within 24h)"}

    logger.info("[media_api] Daily yt-dlp update check triggered...")
    record_ytdlp_update_checked()

    # 1. Try Media API update endpoint if online
    if check_media_api_online():
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(f"{MEDIA_API_BASE}/api/v1/update-ytdlp?force=true")
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"[media_api] Media API updated yt-dlp: {data}")
                    return data
        except Exception as e:
            logger.warning(f"[media_api] Media API update-ytdlp failed ({e}); trying local yt-dlp update")

    # 2. Local fallback update
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        return {"status": "error", "error": "No yt-dlp binary found to update"}

    try:
        proc = subprocess.run(
            [str(ytdlp_bin), "-U"],
            capture_output=True,
            text=True,
            timeout=45.0,
        )
        out = (proc.stdout + "\n" + proc.stderr).strip()
        logger.info(f"[media_api] Local yt-dlp -U completed (code {proc.returncode}): {out}")
        return {"status": "success", "output": out}
    except Exception as e:
        logger.warning(f"[media_api] Local yt-dlp update failed: {e}")
        return {"status": "error", "error": str(e)}


_ONLINE_CACHE_TTL = 10.0
_online_cache: tuple[float, bool] = (0.0, False)


DEFAULT_MEDIA_API_BASES = ("http://127.0.0.1:8080", "http://localhost:8080")
_state_cache: tuple[float, str] = (0.0, "unknown")


def media_api_state(force: bool = False, run=None) -> str:
    """Media API's state from systemd, without connecting to it.

    "running" (the service is up), "ready" (only its socket listens: the first request
    starts it), "down" (neither), or "unknown" (no systemd here, or Media API isn't the
    local socket-activated one). Media API is socket-activated, so an HTTP "is it up?"
    probe *starts* it, and one that lands in its ~1 s shutdown starts it straight back
    up. Asking systemd answers the same question and wakes nothing.
    """
    global _state_cache
    if MEDIA_API_BASE.rstrip("/") not in DEFAULT_MEDIA_API_BASES:
        return "unknown"
    checked_at, cached = _state_cache
    now = time.monotonic()
    if not force and now - checked_at < _ONLINE_CACHE_TTL:
        return cached
    try:
        out = (run or subprocess.run)(["systemctl", "is-active", "media-api.service", "media-api.socket"],
                  capture_output=True, text=True, timeout=3).stdout.split()
        service, socket = (out + ["", ""])[:2]
        state = "running" if service == "active" else "ready" if socket == "active" else "down"
    except (OSError, subprocess.SubprocessError):
        state = "unknown"
    _state_cache = (now, state)
    return state


def check_media_api_online(force: bool = False) -> bool:
    """Returns True if Media API on port 8080 will answer.

    On the workstation this asks systemd (media_api_state): "ready" counts as online,
    because the first real request starts it. Only where systemd can't tell (tests, a
    custom MEDIA_API_URL, no systemd) does it fall back to an HTTP probe, cached for
    _ONLINE_CACHE_TTL seconds because an uncached miss costs the full 1.5s timeout.
    """
    state = media_api_state(force)
    if state != "unknown":
        return state in ("running", "ready")

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


DOWNLOAD_TIMEOUT_S = float(os.getenv("SPEEDMAN_DOWNLOAD_TIMEOUT", "1800"))


def validate_ingest_url(url: str) -> str:
    """Scheme check, shared by the route (fail fast with a 400) and the downloader
    (never trust that the route ran)."""
    url = str(url or "").strip().strip("'\x22<>")
    if not url:
        raise ValueError("URL cannot be empty")
    if url.startswith(("-", "/")):
        raise ValueError("Invalid URL scheme")
    if re.match(r"^(?:www\.)?(?:youtube\.com|youtu\.be|m\.youtube\.com)/", url, re.IGNORECASE):
        url = "https://" + url
    if not _HTTP_URL_RE.match(url):
        raise ValueError("Only http:// and https:// URLs can be ingested")
    return url


def extract_audio_from_url(url: str, on_progress=None) -> Path:
    """Extracts audio from a YouTube or web media URL.

    Delegates to Media API when it is up, and only falls back to a local yt-dlp. That
    ordering matters: YouTube breaks extractors constantly, and Media API's copy
    auto-updates (`auto_update_ytdlp`), while a second binary installed beside it would
    silently rot until the day someone needed it. There is no local yt-dlp on this
    machine today, so Media API is in practice the only path.

    `on_progress` receives dicts shaped like the pipeline's, so a queued job can report
    download progress through the same channel it reports compression progress.
    """
    # yt-dlp takes the URL as a positional argument, so a value beginning with "-" would
    # be parsed as an option (--exec, --config-location, ...). The scheme check plus the
    # "--" terminator in the fallback keep caller input out of yt-dlp's option parser.
    url = validate_ingest_url(url)

    # Check once a day and auto-update yt-dlp when called
    try:
        ensure_ytdlp_updated(force=False)
    except Exception as e:
        logger.warning(f"[media_api] Daily yt-dlp update check failed: {e}")

    if check_media_api_online():
        try:
            return _download_via_media_api(url, on_progress=on_progress)
        except Exception as e:
            logger.warning(f"[media_api] download via Media API failed ({e}); trying local yt-dlp")

    return _download_via_local_ytdlp(url)


def _download_via_media_api(url: str, on_progress=None) -> Path:
    """Queue a download on Media API and wait for the file."""
    logger.info(f"[media_api] Delegating download to Media API: {url}")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{MEDIA_API_BASE}/api/v1/download",
            json={"url": url, "audio_only": True, "format": "wav"},
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Media API rejected the download ({resp.status_code}): {resp.text[:300]}")
        job_id = resp.json().get("job_id")
        if not job_id:
            raise RuntimeError(f"Media API returned no job_id: {resp.text[:200]}")

        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_S
        status = {}
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"download timed out after {DOWNLOAD_TIMEOUT_S:.0f}s")
            try:
                poll_resp = client.get(f"{MEDIA_API_BASE}/api/v1/status/{job_id}")
                if poll_resp.status_code == 200:
                    status = poll_resp.json()
                else:
                    time.sleep(1.5)
                    continue
            except Exception:
                time.sleep(1.5)
                continue

            state = str(status.get("status", "")).lower()

            if on_progress:
                on_progress({
                    "stage": "downloading",
                    "progress_pct": float(status.get("progress_percent") or 0.0),
                })

            if state == "completed":
                break
            if state in ("failed", "error", "cancelled"):
                raise RuntimeError(f"Media API download {state}: {status.get('error') or 'no detail'}")
            time.sleep(1.5)

    outputs = status.get("output_files") or {}
    for key in ("audio", "media", "video"):
        candidate = outputs.get(key)
        if candidate and Path(candidate).is_file():
            logger.info(f"[media_api] Downloaded {candidate}")
            return Path(candidate)
    raise RuntimeError(f"Media API reported success but produced no readable file: {outputs}")


def download_video_from_url(url: str, on_progress=None) -> tuple[Path, bool]:
    """Download the video (not just audio) for previewing.

    Returns (path, pre_existed). `pre_existed` is True when Media API handed back a file
    that was already on disk before we asked -- it deduplicates by URL, so re-downloading
    something from last week returns the old file untouched. The caller must not delete
    such a file: it is the user's, not ours.
    """
    url = validate_ingest_url(url)

    # Check once a day and auto-update yt-dlp when called
    try:
        ensure_ytdlp_updated(force=False)
    except Exception as e:
        logger.warning(f"[media_api] Daily yt-dlp update check failed: {e}")

    if check_media_api_online():
        try:
            return _download_video_via_media_api(url, on_progress=on_progress)
        except Exception as e:
            logger.warning(f"[media_api] Video download via Media API failed ({e}); trying local yt-dlp")

    return _download_video_via_local_ytdlp(url, on_progress=on_progress)


def _download_video_via_media_api(url: str, on_progress=None) -> tuple[Path, bool]:
    """Download video via Media API."""
    requested_at = time.time()
    logger.info(f"[media_api] Requesting video download via Media API: {url}")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{MEDIA_API_BASE}/api/v1/download",
            json={"url": url, "audio_only": False, "format": "mp4"},
        )
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Media API rejected the download ({resp.status_code}): {resp.text[:300]}")
        job_id = resp.json().get("job_id")
        if not job_id:
            raise RuntimeError(f"Media API returned no job_id: {resp.text[:200]}")

        deadline = time.monotonic() + DOWNLOAD_TIMEOUT_S
        status = {}
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"video download timed out after {DOWNLOAD_TIMEOUT_S:.0f}s")
            try:
                poll_resp = client.get(f"{MEDIA_API_BASE}/api/v1/status/{job_id}")
                if poll_resp.status_code == 200:
                    status = poll_resp.json()
                else:
                    time.sleep(1.5)
                    continue
            except Exception:
                time.sleep(1.5)
                continue

            state = str(status.get("status", "")).lower()
            if on_progress:
                on_progress({"stage": "downloading video",
                             "progress_pct": float(status.get("progress_percent") or 0.0)})
            if state == "completed":
                break
            if state in ("failed", "error", "cancelled"):
                raise RuntimeError(f"Media API video download {state}: {status.get('error') or 'no detail'}")
            time.sleep(1.5)

    outputs = status.get("output_files") or {}
    for key in ("video", "media", "audio"):
        candidate = outputs.get(key)
        if candidate and Path(candidate).is_file():
            path = Path(candidate)
            # A file older than our request was already there; Media API deduplicates by
            # URL and returns the existing copy rather than fetching it again.
            pre_existed = path.stat().st_mtime < requested_at
            logger.info(f"[media_api] Video at {path} (pre_existed={pre_existed})")
            return path, pre_existed
    raise RuntimeError(f"Media API reported success but produced no readable file: {outputs}")


MEDIA_API_VENV_YTDLP = Path("/mnt/d/Workspace/media-api/.venv/bin/yt-dlp")
"""Media API's own copy -- the one with auto_update_ytdlp enabled, and the only yt-dlp
this workstation actually has. Borrowing it beats installing a second binary that would
rot silently until the day it was needed."""


def find_ytdlp() -> Optional[Path]:
    """Locate a usable yt-dlp, preferring the copy Media API keeps updated.

    The fallback below used to search only ~/.local/bin and /usr/local/bin, neither of
    which has it here, so the fallback could never fire and the error told the user to
    install something they already had.
    """
    override = os.getenv("SPEEDMAN_YTDLP")
    if override and Path(override).is_file():
        return Path(override)

    candidates = [
        MEDIA_API_VENV_YTDLP,
        Path.home() / ".whisper-env/bin/yt-dlp",
        Path.home() / ".local/bin/yt-dlp",
        Path("/usr/local/bin/yt-dlp"),
        Path("/usr/bin/yt-dlp"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    on_path = shutil.which("yt-dlp")
    return Path(on_path) if on_path else None


def _download_via_local_ytdlp(url: str) -> Path:
    """Fallback for when Media API is down, using whatever yt-dlp this machine has."""
    ytdlp_bin = find_ytdlp()

    if not ytdlp_bin:
        raise RuntimeError(
            f"Cannot download: Media API ({MEDIA_API_BASE}) is unreachable and no yt-dlp "
            "could be found. Start Media API -- it owns the auto-updating yt-dlp this "
            "workstation uses -- or point SPEEDMAN_YTDLP at a binary."
        )

    ytdlp_bin = str(ytdlp_bin)
    logger.info(f"[media_api] Media API is down; falling back to {ytdlp_bin} for {url}")

    out_template = str(DOWNLOADS_DIR / "%(title).200B.%(ext)s")

    # E-core pinning is a nicety, not a requirement: if the tools are missing, download
    # unpinned rather than failing with FileNotFoundError on taskset.
    prefix: List[str] = []
    if shutil.which("taskset"):
        prefix += ["taskset", "-c", "8-15"]
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]

    js_args: List[str] = []
    if shutil.which("node"):
        js_args = ["--js-runtimes", "node", "--remote-components", "ejs:github"]

    cmd = [
        *prefix,
        ytdlp_bin,
        *js_args,
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
        "--print", "filepath",
        "--",  # end of options; everything after this is a positional URL
        url,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"yt-dlp download failed: {err}")

    # Find the printed path
    lines = [ln.strip() for ln in proc.stdout.strip().split("\n") if ln.strip()]
    for ln in reversed(lines):
        p = Path(ln)
        if p.is_file():
            return p
        wav_p = p.with_suffix(".wav")
        if wav_p.is_file():
            return wav_p

    # Fallback: scan DOWNLOADS_DIR for newest WAV
    wavs = sorted(DOWNLOADS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        return wavs[0]

    raise RuntimeError("Audio was downloaded but target file could not be determined")


def _download_video_via_local_ytdlp(url: str, on_progress=None) -> tuple[Path, bool]:
    """Fallback to download video locally when Media API is unreachable."""
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        raise RuntimeError(
            f"Cannot download video: Media API ({MEDIA_API_BASE}) is unreachable and no yt-dlp "
            "could be found. Start Media API or point SPEEDMAN_YTDLP at a binary."
        )

    ytdlp_bin = str(ytdlp_bin)
    logger.info(f"[media_api] Media API is down; downloading video locally with {ytdlp_bin} for {url}")

    from app import video as video_cache
    video_cache.VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_template = str(video_cache.VIDEO_CACHE_DIR / "%(title).200B_%(id)s.%(ext)s")

    prefix: List[str] = []
    if shutil.which("taskset"):
        prefix += ["taskset", "-c", "8-15"]
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]

    js_args: List[str] = []
    if shutil.which("node"):
        js_args = ["--js-runtimes", "node", "--remote-components", "ejs:github"]

    if on_progress:
        on_progress({"stage": "downloading video", "progress_pct": 10.0})

    cmd = [
        *prefix,
        ytdlp_bin,
        *js_args,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", out_template,
        "--print", "after_move:filepath",
        "--print", "filepath",
        "--",
        url,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"yt-dlp video download failed: {err}")

    lines = [ln.strip() for ln in proc.stdout.strip().split("\n") if ln.strip()]
    for ln in reversed(lines):
        p = Path(ln)
        if p.is_file():
            return p, False
        mp4_p = p.with_suffix(".mp4")
        if mp4_p.is_file():
            return mp4_p, False

    mp4s = sorted(video_cache.VIDEO_CACHE_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if mp4s:
        return mp4s[0], False

    raise RuntimeError("Video was downloaded but target file could not be determined")


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


def get_media_api_job_status(job_id: str) -> Dict[str, Any]:
    """Queries Media API for the status of a background job (STT transcription, download, etc.)."""
    if not check_media_api_online():
        raise RuntimeError("Media API (127.0.0.1:8080) is currently offline")

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{MEDIA_API_BASE}/api/v1/status/{job_id}")
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise RuntimeError(f"Job '{job_id}' not found in Media API")
        err = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise RuntimeError(f"Media API status query failed: {err}")

