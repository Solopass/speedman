"""FastAPI Application for Speedman Speech Engine.

Supports on-demand systemd socket activation, 0-RAM standby, auto-close watchdog,
speech compression (1.01x to 30x), blind A/B comparison sets, interactive Web UI,
asynchronous job queue with cancellation, long-file chunking, Media API integration,
and Windows/WSL path interoperability.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from speedman import io as sio
from speedman.config import PRESETS, build_config
from speedman.pipeline import process
from app import timemap_store, transcript as transcript_sync, media_api
from app.paths import normalize_path, to_windows_path, is_within, safe_stem
from app.range_response import range_stream_file
from app.queue import job_queue, SpeedmanJob
from app.media_api import (
    check_media_api_online,
    get_media_api_health,
    list_media_library,
    extract_audio_from_url,
    ensure_ytdlp_updated,
    send_to_media_api_transcribe,
    validate_ingest_url,
)

# Paths
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = Path("/mnt/d/Audio/Speed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR = OUTPUT_DIR / "comparisons"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

# Durable logging on D:
LOG_FILE = OUTPUT_DIR / "speedman.log"
file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [speedman] %(message)s"))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [speedman] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger("speedman_api")


# --------------------------------------------------------------------------- Paths & Security

ALLOWED_ROOTS = [
    Path("/mnt/d/Audio"),
    Path("/mnt/d/Workspace"),
    Path("/mnt/d/Output"),
    Path("/mnt/d/AI"),
    Path("/mnt/d/OBVLT"),
    Path("/tmp"),
]


def is_path_allowed(target: Path) -> bool:
    """True when target resolves to one of ALLOWED_ROOTS or inside it.

    This governs which files may be read *as compression input*, which is why it spans
    most of D:. It is deliberately NOT used to decide what may be served back over
    HTTP -- see resolve_output_file().
    """
    return any(is_within(target, root) for root in ALLOWED_ROOTS)


def resolve_output_file(filename: str) -> Path:
    r"""Resolve a client-supplied name to a file inside OUTPUT_DIR, or raise 404.

    Validating against ALLOWED_ROOTS here would be a directory traversal: those roots
    cover D:\Workspace, D:\AI and D:\OBVLT, so ".." segments (or their percent-encoded
    form, which the client never collapses) let a caller stream any file in those trees.
    Everything Speedman produces lives under OUTPUT_DIR, so that is the only root worth
    honouring.
    """
    clean = str(filename).strip("/\\")
    out_root = OUTPUT_DIR.resolve()

    target = (out_root / clean).resolve()
    if target.is_file() and is_within(target, out_root):
        return target

    # Bare-name lookup, so /api/v1/audio/<name> still finds files in subfolders
    # (comparison sets, downloads). rglob is rooted at OUTPUT_DIR, so it cannot escape.
    name_only = Path(clean).name
    if name_only:
        for candidate in sorted(out_root.rglob(name_only)):
            if candidate.is_file() and is_within(candidate, out_root):
                return candidate

    raise HTTPException(status_code=404, detail=f"Audio file '{clean}' not found")


# --------------------------------------------------------------------------- Session & Watchdog

class SessionManager:
    """Tracks active browser tabs/clients and active audio processing jobs."""
    def __init__(self):
        self.active_jobs: int = 0
        self.sessions: dict[str, float] = {}
        self.last_active_time: float = time.time()
        self.auto_close: bool = os.getenv("AUTO_CLOSE", "true").lower() in ("true", "1", "yes")
        self.idle_timeout: int = int(os.getenv("IDLE_TIMEOUT", "60"))
        # How long a client counts as present after its last heartbeat. This MUST exceed
        # the worst-case heartbeat interval, and the client beats every 5s only while
        # visible: browsers throttle background-tab timers to about once a minute. At the
        # old 12s a backgrounded tab was forgotten within seconds and the watchdog shut
        # the service down underneath an open window. Closing a tab does not rely on this
        # -- beforeunload sends an explicit disconnect.
        self.session_ttl: float = float(os.getenv("CLIENT_SESSION_TTL", "90"))

    def mark_activity(self):
        self.last_active_time = time.time()

    def heartbeat(self, session_id: str):
        self.sessions[session_id] = time.time()
        self.last_active_time = time.time()

    def disconnect(self, session_id: str):
        self.sessions.pop(session_id, None)
        self.last_active_time = time.time()

    def active_client_count(self) -> int:
        now = time.time()
        self.sessions = {sid: ts for sid, ts in self.sessions.items()
                         if now - ts < self.session_ttl}
        return len(self.sessions)

    def inc_job(self):
        self.active_jobs += 1
        self.last_active_time = time.time()

    def dec_job(self):
        self.active_jobs = max(0, self.active_jobs - 1)
        self.last_active_time = time.time()

    def should_shutdown(self) -> bool:
        if not self.auto_close or self.idle_timeout <= 0:
            return False
        if self.active_jobs > 0 or job_queue.active_count() > 0:
            self.last_active_time = time.time()
            return False
        if self.active_client_count() > 0:
            self.last_active_time = time.time()
            return False
        idle = time.time() - self.last_active_time
        return idle >= self.idle_timeout


jobs = SessionManager()


async def idle_watchdog():
    """Monitors client sessions and active jobs. Shuts down after idle_timeout when browser is closed."""
    if not jobs.auto_close:
        logger.info("[watchdog] Auto-close disabled.")
        return
    logger.info(f"[watchdog] Auto-close active: counts down {jobs.idle_timeout}s ONLY AFTER you exit the browser app.")
    while True:
        try:
            await asyncio.sleep(5)
            if jobs.should_shutdown():
                idle = int(time.time() - jobs.last_active_time)
                logger.info(f"[watchdog] Browser app has been closed for {idle}s with 0 active jobs. Releasing RAM...")
                os.kill(os.getpid(), signal.SIGTERM)
                break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[watchdog] Watchdog error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Leftover videos from a session that never came back. Only touches Speedman's own
    # cache, and only entries older than a day.
    try:
        from app import video as video_cache

        swept = video_cache.sweep()
        if swept["deleted"] or swept["forgotten"]:
            logger.info(f"[startup] video cache sweep: {swept}")
    except Exception as e:
        logger.warning(f"[startup] video cache sweep skipped: {e}")

    async def _startup_ytdlp_check():
        await asyncio.sleep(3)
        try:
            from app.media_api import ensure_ytdlp_updated, should_check_ytdlp_update
            if should_check_ytdlp_update():
                await asyncio.to_thread(ensure_ytdlp_updated, force=False)
        except Exception as e:
            logger.warning(f"[startup] background yt-dlp update check failed: {e}")

    ytdlp_task = asyncio.create_task(_startup_ytdlp_check())
    watchdog = asyncio.create_task(idle_watchdog())
    yield
    watchdog.cancel()
    ytdlp_task.cancel()
    try:
        await watchdog
    except asyncio.CancelledError:
        pass
    try:
        await ytdlp_task
    except asyncio.CancelledError:
        pass


# --------------------------------------------------------------------------- FastAPI App

app = FastAPI(
    title="Speedman Speech Engine API",
    description="Speech-aware dynamic time compression microservice that preserves intelligibility at 5x–6x speeds.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# The Studio UI is served from this same origin, so it needs no CORS grant at all. A
# wildcard therefore only grants access to *other* pages the user happens to be visiting,
# which on a loopback service with file-read and shutdown routes is a liability. Override
# with SPEEDMAN_CORS_ORIGINS="http://host:port,..." if another local client needs in.
_cors_env = os.getenv("SPEEDMAN_CORS_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://127.0.0.1:8081", "http://localhost:8081", "null"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_activity(request: Request, call_next):
    if not request.url.path.endswith("/health"):
        jobs.mark_activity()
    return await call_next(request)


# --------------------------------------------------------------------------- Models

class CompressRequest(BaseModel):
    input_path: str = Field(..., description="Local path to audio file on workstation (Windows or WSL format)")
    speed: float = Field(5.0, ge=1.01, le=30.0, description="Speed multiplier (e.g. 5.0 for 5x)")
    preset: str = Field("fast", description="Preset: natural | fast | audiophile | aggressive | max")
    uniform: bool = Field(False, description="Uniform constant-rate stretch (control mode)")
    format: str = Field("wav", description="Audio output format: wav | mp3 | m4a | flac")
    output_filename: Optional[str] = None
    auto_obsidian: bool = Field(False, description="Automatically generate Obsidian note upon completion")


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="YouTube or web media URL to extract and compress")
    speed: float = Field(5.0, ge=1.01, le=30.0)
    preset: str = Field("fast")
    uniform: bool = Field(False)
    format: str = Field("wav", description="Audio output format: wav | mp3 | m4a | flac")
    include_video: bool = Field(
        False,
        description="Also fetch the video so it can be watched. Cached temporarily and "
                    "discarded unless saved; the audio and transcript are always kept.",
    )


class TranscribeRequest(BaseModel):
    engine: str = Field("parakeet", description="Speech-to-text engine: parakeet or whisper")
    source_path: Optional[str] = Field(
        None,
        description="The ORIGINAL audio to transcribe. Compression destroys ASR accuracy "
                    "(measured: 150 words at 1x, 11 words of nonsense at 5x), so "
                    "transcription always runs on the source. Compression results carry "
                    "this as 'source_path'; omit it and the source is derived from the "
                    "output filename.",
    )
    sync_to_output: Optional[str] = Field(
        None,
        description="A compressed filename to sync the finished transcript against. The "
                    "1x timestamps are warped through that output's time map and written "
                    "as <stem>.vtt and <stem>.synced.json beside it.",
    )


class TranscribePipelineRequest(BaseModel):
    output_name: str = Field(..., description="Compressed filename to sync against")
    source_path: Optional[str] = Field(None, description="The original audio source path")
    engine: str = Field("parakeet", description="Speech-to-text engine: parakeet or whisper")
    wait: bool = Field(False, description="If True, block until Media API transcription completes, then run sync")
    timeout_s: float = Field(300.0, description="Max seconds to wait if wait=True")
    auto_chapters: bool = Field(True, description="Auto-generate and cache semantic chapters on sync completion")
    auto_obsidian: bool = Field(False, description="Auto-export summary note to Obsidian on completion")
    obsidian_folder: str = Field("Summaries", description="Folder in 1Notebook for Obsidian note")
    use_llm: bool = Field(True, description="Whether to use local LLM for summary/chapters")


class CompareRequest(BaseModel):
    input_path: str = Field(..., description="Local path to audio file on workstation")
    speeds: List[float] = Field(default_factory=lambda: [4.0, 5.0, 6.0], description="List of target speeds to compare")


class HeartbeatRequest(BaseModel):
    session_id: str


class ListeningSessionRequest(BaseModel):
    speeds: List[float] = Field(default_factory=lambda: [3.0, 4.0, 5.0, 6.0, 8.0])
    trials_per_speed: int = Field(6, ge=1, le=50)
    excerpt_s: float = Field(30.0, ge=3.0, le=120.0)
    clips: Optional[List[str]] = Field(None, description="Clip labels; null means all available")


class ListeningVerdictRequest(BaseModel):
    trial_index: int
    choice: str = Field(..., description="a | b | none")
    followed: str = Field(..., description="both | picked | neither")


class OpenFolderRequest(BaseModel):
    folder: str = Field(..., description="Folder identifier: audio | videos | source_audio | video_cache")


class ObsidianExportRequest(BaseModel):
    output_name: str = Field(..., description="Target compressed audio or transcript stem")
    folder: str = Field("Summaries", description="Subfolder in 1Notebook (e.g. Summaries, Digests)")
    note_title: Optional[str] = Field(None, description="Custom note title, defaults to stem")
    include_summary: bool = Field(True, description="Generate executive summary & key takeaways")
    include_highlights: bool = Field(True, description="Include 5s marked quotes and bookmarks")
    include_transcript: bool = Field(True, description="Include collapsible transcript outline")
    custom_tags: Optional[List[str]] = Field(None, description="Additional Obsidian tags")
    bookmarks: Optional[List[Dict[str, Any]]] = Field(None, description="Client-side bookmarks with timestamps and notes")
    use_llm: bool = Field(True, description="Attempt local AI synthesis on :11440 with graceful fallback")
    embed_audio: bool = Field(False, description="Embed audio player ![[<audio>]] in note")
    deep_link_timestamps: bool = Field(False, description="Format timestamps as obsidian:// deep-links")


class DailyDigestExportRequest(BaseModel):
    quotes: List[Dict[str, Any]] = Field(..., description="Quotes and bookmarks to append")
    source_title: str = Field(..., description="Title of the source audio")
    folder: str = Field("Digests", description="Target 1Notebook subfolder (defaults to Digests)")
    date_str: Optional[str] = Field(None, description="Optional target date YYYY-MM-DD")


class CleanCacheRequest(BaseModel):
    days: float = Field(7.0, ge=0.0, description="Age threshold in days")
    all_files: bool = Field(False, description="Purge all cache files regardless of age")


class OpenObsidianNoteRequest(BaseModel):
    path: str = Field(..., description="WSL or Windows path to the note")


class GenerateChaptersRequest(BaseModel):
    use_llm: bool = Field(True, description="Attempt local AI chapter labeling via :11440 with algorithmic fallback")
    force_regenerate: bool = Field(False, description="Ignore cached chapters and recompute")


class EditCueRequest(BaseModel):
    line_index: int = Field(..., description="0-indexed cue line index to modify")
    new_text: str = Field(..., description="Replacement text for the cue")


class FindReplaceRequest(BaseModel):
    find: str = Field(..., description="Text or pattern to search for")
    replace: str = Field(..., description="Replacement text")
    match_case: bool = Field(False, description="Case-sensitive search if True")


# --------------------------------------------------------------------------- Core Routes

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "speedman",
        "version": "1.1.0",
        "active_jobs": jobs.active_jobs + job_queue.active_count(),
        "active_clients": jobs.active_client_count(),
        "idle_seconds": int(time.time() - jobs.last_active_time),
        "output_dir": str(OUTPUT_DIR),
        "windows_output_dir": to_windows_path(OUTPUT_DIR),
        "media_api_online": check_media_api_online(),
    }


@app.post("/api/v1/heartbeat")
def client_heartbeat(req: HeartbeatRequest):
    jobs.heartbeat(req.session_id)
    return {"status": "ok", "active_clients": jobs.active_client_count()}


@app.post("/api/v1/client-disconnect")
async def client_disconnect(request: Request):
    try:
        data = await request.json()
        jobs.disconnect(data.get("session_id", ""))
    except Exception:
        try:
            raw = (await request.body()).decode()
            data = json.loads(raw)
            jobs.disconnect(data.get("session_id", ""))
        except Exception:
            pass
    return {"status": "ok", "active_clients": jobs.active_client_count()}


@app.post("/api/v1/shutdown")
async def shutdown():
    import threading
    logger.info("[api] Shutdown requested via POST /api/v1/shutdown")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return {"status": "shutting_down", "message": "Speedman service stopping cleanly"}


@app.get("/api/v1/presets")
def list_presets():
    return {
        name: {
            "name": name,
            "post": cfg["post"],
            "ratemap": cfg["ratemap"],
        }
        for name, cfg in PRESETS.items()
    }


def require_output_format(fmt: str) -> str:
    """Normalise a requested output format, or 400.

    Never silently downgrades to wav: the old fallback wrote a .wav file while still
    reporting back whatever format the caller had asked for.
    """
    try:
        return sio.normalize_output_format(fmt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Speedman names outputs "<stem>_<speed>x_<preset>[_uniform].<ext>". Recovering the stem
# is what lets a transcription request find the audio the output was made from.
_OUTPUT_NAME_RE = re.compile(r"^(?P<stem>.+?)_(?P<speed>[\d.]+)x_(?P<preset>[a-z]+)(?:_uniform)?$")

SOURCE_SEARCH_DIRS = [
    Path("/mnt/d/Output/Audio"),
    Path("/mnt/d/Output/Videos"),
    OUTPUT_DIR / "downloads",
    OUTPUT_DIR / "video-cache",
]


def looks_like_speedman_output(path: Path) -> bool:
    """True when this is one of our compressed outputs rather than a source."""
    return bool(_OUTPUT_NAME_RE.match(safe_stem(path))) and is_within(path, OUTPUT_DIR)


def find_source_for_output(output_name: str) -> Optional[Path]:
    """Locate the audio a compressed output was produced from.

    Best-effort: compression results carry `source_path`, and callers should pass it.
    This exists for the case where only a filename is to hand.
    """
    match = _OUTPUT_NAME_RE.match(safe_stem(output_name))
    if not match:
        return None
    stem = match.group("stem")
    for directory in SOURCE_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob(f"{stem}.*")):
            if candidate.is_file() and not looks_like_speedman_output(candidate):
                return candidate
    return None


def find_transcript_for_source(source: Optional[Path], output_name: Optional[str] = None) -> Optional[Path]:
    """Search for the transcript JSON produced by Media API or present in library.

    Media API saves transcripts at `<audio_path>.with_suffix('.transcript.json')`. For videos,
    it extracts audio to /mnt/d/Output/Audio/<stem>.mp3 and saves the transcript to
    /mnt/d/Output/Audio/<stem>.transcript.json, so looking only beside `source` fails when
    `source` is a video or resides in downloads/video-cache.
    """
    candidates: list[Path] = []

    if source is not None:
        source_p = Path(source)
        src_stem = safe_stem(source_p)
        # 1. Beside source
        candidates.append(source_p.with_suffix(".transcript.json"))
        candidates.append(source_p.parent / f"{src_stem}.transcript.json")
        candidates.append(source_p.with_suffix(".json"))
        candidates.append(source_p.parent / f"{src_stem}.json")
        # 2. In Media API audio / video output dirs
        candidates.append(media_api.MEDIA_API_AUDIO_DIR / f"{source_p.stem}.transcript.json")
        candidates.append(media_api.MEDIA_API_AUDIO_DIR / f"{src_stem}.transcript.json")
        candidates.append(media_api.MEDIA_API_AUDIO_DIR / f"{source_p.name}.transcript.json")
        candidates.append(media_api.MEDIA_API_VIDEO_DIR / f"{source_p.stem}.transcript.json")
        candidates.append(media_api.MEDIA_API_VIDEO_DIR / f"{src_stem}.transcript.json")
        # 3. In downloads dir
        candidates.append(media_api.DOWNLOADS_DIR / f"{source_p.stem}.transcript.json")
        candidates.append(media_api.DOWNLOADS_DIR / f"{src_stem}.transcript.json")
        # 4. In video-cache dir
        try:
            from app import video as video_cache
            candidates.append(video_cache.VIDEO_CACHE_DIR / f"{source_p.stem}.transcript.json")
            candidates.append(video_cache.VIDEO_CACHE_DIR / f"{src_stem}.transcript.json")
        except Exception:
            pass
        # 5. In OUTPUT_DIR
        candidates.append(OUTPUT_DIR / f"{source_p.stem}.transcript.json")
        candidates.append(OUTPUT_DIR / f"{src_stem}.transcript.json")

    if output_name:
        out_stem = safe_stem(output_name)
        match = _OUTPUT_NAME_RE.match(out_stem)
        base_stem = match.group("stem") if match else out_stem
        candidates.append(media_api.MEDIA_API_AUDIO_DIR / f"{base_stem}.transcript.json")
        candidates.append(media_api.MEDIA_API_VIDEO_DIR / f"{base_stem}.transcript.json")
        candidates.append(media_api.DOWNLOADS_DIR / f"{base_stem}.transcript.json")
        candidates.append(OUTPUT_DIR / f"{base_stem}.transcript.json")
        candidates.append(OUTPUT_DIR / f"{out_stem}.transcript.json")

    for cand in candidates:
        try:
            cand_res = cand.resolve()
            if cand_res.is_file() and is_path_allowed(cand_res):
                return cand_res
        except Exception:
            continue
    return None



def get_video_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".webm":
        return "video/webm"
    if ext == ".mkv":
        return "video/x-matroska"
    return "video/mp4"


def get_audio_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    # Synced transcripts live beside the audio and are served by the same route. A .vtt
    # handed back as audio/wav is silently ignored by <track>.
    if ext == ".vtt":
        return "text/vtt"
    if ext in (".srt", ".txt"):
        return "text/plain; charset=utf-8"
    if ext == ".json":
        return "application/json"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext in (".m4a", ".mp4", ".aac"):
        return "audio/mp4"
    if ext == ".flac":
        return "audio/flac"
    if ext in (".ogg", ".opus"):
        return "audio/ogg"
    return "audio/wav"


@app.get("/api/v1/audio/{filename:path}")
def stream_audio(filename: str, request: Request):
    target = resolve_output_file(filename)
    return range_stream_file(target, request, media_type=get_audio_media_type(target))


# --------------------------------------------------------------------------- Synchronous Compression

def _run_compression(
    src_path: Path,
    stem: str,
    speed: float,
    preset: str,
    uniform: bool,
    format: str = "wav",
    trusted_path: bool = False,
    auto_obsidian: bool = False,
) -> dict[str, Any]:
    """trusted_path=True for server-created temp files. Their location is ours, not the
    caller's, so checking them against ALLOWED_ROOTS can only misfire -- every upload
    would start 403-ing if TMPDIR ever moved off /tmp."""
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    fmt = require_output_format(format)

    if not trusted_path and not is_path_allowed(src_path):
        raise HTTPException(status_code=403, detail=f"Input path '{src_path}' is outside permitted workstation roots")

    if not src_path.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {src_path}")

    t_start = time.perf_counter()
    cfg = build_config(speed=speed, preset=preset, backend="rubberband", uniform=uniform)
    y = sio.load(src_path, cfg.sample_rate)
    in_dur = len(y) / cfg.sample_rate

    res = process(y, cfg.sample_rate, cfg)

    out_name = f"{stem}_{speed:g}x_{preset}{'_uniform' if uniform else ''}.{fmt}"
    out_path = OUTPUT_DIR / out_name
    sio.save(out_path, res.audio, res.sr)
    out_dur = len(res.audio) / res.sr
    # Keep the map so a 1x transcript can be synced to this output later.
    timemap_store.save_quietly(out_name, res.time_map, speed, res.sr)

    try:
        from speedman.post import compute_waveform_peaks
        peaks = compute_waveform_peaks(res.audio, num_bins=120)
        (OUTPUT_DIR / f"{safe_stem(out_name)}.peaks.json").write_text(
            json.dumps({"peaks": peaks, "filename": out_name}), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Failed to cache waveform peaks: {e}")

    elapsed = time.perf_counter() - t_start

    result = {
        "status": "success",
        "filename": out_name,
        "output_path": str(out_path),
        "windows_output_path": to_windows_path(out_path),
        # Provenance: transcription must run on this, never on the output above.
        "source_path": str(src_path),
        "windows_source_path": to_windows_path(src_path),
        "audio_url": f"/api/v1/audio/{out_name}",
        "speed": speed,
        "preset": preset,
        "uniform": uniform,
        "format": fmt,
        "input_duration_s": round(in_dur, 2),
        "output_duration_s": round(out_dur, 2),
        "compression_ratio": round(in_dur / max(out_dur, 0.001), 2),
        "silence_fraction": res.notes.get("silence_fraction", 0.0),
        "effective_speech_rate": res.notes.get("effective_speech_rate", speed),
        "estimated_speech_rate": res.notes.get("estimated_speech_rate", speed),
        # >0 means the map is running at its ceiling and has stopped being non-uniform.
        "rate_clamped_fraction": res.notes.get("rate_clamped_fraction", 0.0),
        "processing_time_s": round(elapsed, 2),
        "timings": res.timings,
        "chunked": res.notes.get("chunked", False),
    }

    if auto_obsidian:
        try:
            from app import obsidian
            src_transcript = find_transcript_for_source(src_path, out_name)
            segments = []
            if src_transcript and src_transcript.is_file():
                try:
                    tdata = json.loads(src_transcript.read_text(encoding="utf-8"))
                    segments = tdata.get("segments", [])
                except Exception:
                    pass
            summary_data = None
            if segments:
                summary_data = obsidian.generate_note_summary(segments, use_llm=True)
            md_content = obsidian.format_obsidian_markdown(
                title=stem,
                source_name=src_path.name,
                duration_s=in_dur,
                speed=speed,
                summary_data=summary_data,
                segments=segments,
                include_transcript=bool(segments),
            )
            obsidian_note = obsidian.save_obsidian_note(
                note_title=stem,
                content=md_content,
                folder="Summaries",
            )
            result["obsidian_note"] = obsidian_note
            logger.info(f"[compress/obsidian] Saved auto-note to {obsidian_note.get('path')}")
        except Exception as e:
            logger.warning(f"[compress/obsidian] Auto-note creation failed: {e}")

    return result


@app.post("/api/v1/compress")
async def compress_audio_multipart(
    file: Optional[UploadFile] = File(None),
    input_path: Optional[str] = Form(None),
    speed: float = Form(5.0),
    preset: str = Form("fast"),
    uniform: bool = Form(False),
    format: str = Form("wav"),
    auto_obsidian: bool = Form(False),
):
    jobs.inc_job()
    temp_in = None
    try:
        if file is not None:
            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                temp_in = Path(tf.name)
                shutil.copyfileobj(file.file, tf)
            src_path = temp_in
            stem = Path(file.filename or "audio").stem
        elif input_path:
            src_path = normalize_path(input_path).resolve()
            stem = src_path.stem
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'input_path'")

        return _run_compression(
            src_path, stem, speed, preset, uniform, format=format, trusted_path=temp_in is not None, auto_obsidian=auto_obsidian
        )
    finally:
        if temp_in and temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass
        jobs.dec_job()


@app.post("/api/v1/compress/json")
async def compress_audio_json(req: CompressRequest):
    jobs.inc_job()
    try:
        src_path = normalize_path(req.input_path).resolve()
        stem = src_path.stem
        return _run_compression(src_path, stem, req.speed, req.preset, req.uniform, format=req.format, auto_obsidian=req.auto_obsidian)
    finally:
        jobs.dec_job()


# --------------------------------------------------------------------------- Asynchronous Job Queue

@app.post("/api/v1/jobs/compress")
async def queue_compress_job(req: CompressRequest):
    """Queues a non-blocking background compression job."""
    src_path = normalize_path(req.input_path).resolve()
    if not is_path_allowed(src_path):
        raise HTTPException(status_code=403, detail=f"Path '{src_path}' is outside permitted workstation roots")
    if not src_path.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {src_path}")

    fmt = require_output_format(req.format)
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{req.preset}'. Available: {list(PRESETS.keys())}")

    job = job_queue.submit(
        input_path=src_path,
        speed=req.speed,
        preset=req.preset,
        uniform=req.uniform,
        output_dir=OUTPUT_DIR,
        output_format=fmt,
        auto_obsidian=req.auto_obsidian,
    )
    return {
        "status": "queued",
        "job_id": job.job_id,
        "input_filename": src_path.name,
        "speed": req.speed,
        "preset": req.preset,
        "format": job.output_format,
    }


@app.get("/api/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """Fetches real-time status, progress %, stage, and measured ETA for a job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job_route(job_id: str):
    """Cancels a queued or running job, immediately stopping processing on E-cores."""
    ok = job_queue.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"status": "cancellation_requested", "job_id": job_id}


@app.get("/api/v1/jobs")
def list_recent_jobs():
    """Lists recent compression jobs and their status."""
    return job_queue.list_jobs(limit=25)


@app.get("/api/v1/outputs")
def list_saved_outputs(limit: int = 50):
    """Lists saved compressed audio outputs in OUTPUT_DIR and their transcript status."""
    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
    items = []
    try:
        for p in OUTPUT_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in audio_exts and not p.name.startswith("."):
                stat = p.stat()
                stem = safe_stem(p)
                synced_json = p.with_name(f"{stem}.synced.json")
                vtt = p.with_name(f"{stem}.vtt")
                chapters_json = p.with_name(f"{stem}.chapters.json")
                has_transcript = synced_json.is_file()
                has_vtt = vtt.is_file()
                has_chapters = chapters_json.is_file()

                speed_match = re.search(r"_(\d+(?:\.\d+)?)x_", stem)
                speed = float(speed_match.group(1)) if speed_match else None

                items.append({
                    "filename": p.name,
                    "stem": stem,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "format": p.suffix.lower().lstrip("."),
                    "speed": speed,
                    "has_transcript": has_transcript,
                    "has_vtt": has_vtt,
                    "has_chapters": has_chapters,
                    "audio_url": f"/api/v1/audio/{p.name}",
                    "transcript_url": f"/api/v1/transcript/{p.name}" if has_transcript else None,
                })
        items.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception as e:
        logger.warning(f"Error listing saved outputs: {e}")
    return items[:limit]


@app.delete("/api/v1/outputs/{filename}")
def delete_saved_output(filename: str):
    """Safely deletes a compressed audio output and all its companion transcript and chapter files."""
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = OUTPUT_DIR / filename
    if not is_within(target, OUTPUT_DIR):
        raise HTTPException(status_code=403, detail="Path traversal forbidden")

    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found")

    stem = safe_stem(target)
    deleted_files = []

    # Audio file
    try:
        target.unlink()
        deleted_files.append(target.name)
    except Exception as e:
        logger.error(f"Failed to delete audio file {target}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not delete audio file: {e}")

    # Companion files
    companion_patterns = [
        f"{stem}.synced.json",
        f"{stem}.transcript.json",
        f"{stem}.vtt",
        f"{stem}.source.vtt",
        f"{stem}.srt",
        f"{stem}.source.srt",
        f"{stem}.txt",
        f"{stem}.chapters.json",
        f"{stem}.peaks.json",
    ]
    for pattern in companion_patterns:
        comp = OUTPUT_DIR / pattern
        if comp.is_file() and is_within(comp, OUTPUT_DIR):
            try:
                comp.unlink()
                deleted_files.append(comp.name)
            except Exception as e:
                logger.warning(f"Failed to delete companion file {comp}: {e}")

    return {
        "status": "deleted",
        "filename": filename,
        "deleted_files": deleted_files,
    }


# --------------------------------------------------------------------------- Media API Integration

@app.get("/api/v1/media/status")
def media_api_status():
    """Checks whether Media API (127.0.0.1:8080) is online."""
    online = check_media_api_online()
    health_data = get_media_api_health() if online else None
    return {
        "online": online,
        "url": "http://127.0.0.1:8080",
        "health": health_data,
    }


@app.get("/api/v1/media/library")
def media_api_library():
    """Lists audio and video files available in Media API's output directories."""
    return list_media_library(limit=40)


@app.post("/api/v1/tools/update-ytdlp")
async def trigger_ytdlp_update(force: bool = Query(True, description="Force update check even if checked recently")):
    """Checks and updates yt-dlp on the workstation."""
    return await asyncio.to_thread(ensure_ytdlp_updated, force=force)


@app.post("/api/v1/tools/open-folder")
def open_workstation_folder(req: OpenFolderRequest):
    """Opens a workstation media directory in Windows Explorer."""
    folder_map = {
        "audio": OUTPUT_DIR,
        "videos": Path("/mnt/d/Output/Videos"),
        "source_audio": Path("/mnt/d/Output/Audio"),
        "video_cache": Path("/mnt/d/Audio/Speed/video-cache"),
        "1notebook": Path("/mnt/d/OBVLT/1Notebook"),
        "obsidian": Path("/mnt/d/OBVLT"),
    }
    target = folder_map.get(req.folder.lower().strip())
    if target is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown folder '{req.folder}'. Available: {list(folder_map.keys())}",
        )

    target.mkdir(parents=True, exist_ok=True)
    win_path = to_windows_path(target)

    explorer_path = Path("/mnt/c/WINDOWS/explorer.exe")
    if not explorer_path.exists():
        explorer_path = Path("/mnt/c/Windows/explorer.exe")

    try:
        if explorer_path.exists():
            subprocess.Popen([str(explorer_path), win_path])
        else:
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
        return {"status": "success", "folder": req.folder, "path": win_path}
    except Exception as e:
        logger.error(f"[tools] Failed to open folder '{win_path}' in Explorer: {e}")
        raise HTTPException(status_code=500, detail=f"Could not open folder in Explorer: {e}")


@app.post("/api/v1/ingest/url")
async def ingest_url_and_compress(req: IngestUrlRequest):
    """Queues a YouTube/web URL for download and compression.

    The download runs inside the job, not inside this request: a two-hour podcast would
    otherwise hold the HTTP connection open for minutes with no progress to show. The
    client polls /api/v1/jobs/{id} and sees a 'downloading' stage before 'stretching'.
    """
    fmt = require_output_format(req.format)
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{req.preset}'. Available: {list(PRESETS.keys())}")

    # Validate the URL here so an obviously bad one fails fast with a 400 rather than
    # surfacing later as a failed job.
    try:
        validate_ingest_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = job_queue.submit(
        source_url=req.url,
        speed=req.speed,
        preset=req.preset,
        uniform=req.uniform,
        output_dir=OUTPUT_DIR,
        output_format=fmt,
        include_video=req.include_video,
    )
    return {
        "status": "queued",
        "job_id": job.job_id,
        "source_url": req.url,
        "speed": req.speed,
        "preset": req.preset,
        "format": job.output_format,
        "include_video": req.include_video,
    }


# --------------------------------------------------------------------------- Video cache
#
# Watch a downloaded video, then keep it or let it go. Default is to discard: the audio
# and transcript are the point, the video is scratch. See app/video.py for why deletion
# is confined to Speedman's own cache directory.

@app.get("/api/v1/video")
def list_cached_videos():
    from app import video as video_cache

    return video_cache.list_cached()


@app.get("/api/v1/video/{video_id}")
def get_cached_video(video_id: str):
    from app import video as video_cache

    try:
        return video_cache.get(video_id).public()
    except video_cache.VideoNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/video/{video_id}/stream")
def stream_cached_video(video_id: str, request: Request):
    """Range-served so the browser can seek without downloading the whole file."""
    from app import video as video_cache

    try:
        entry = video_cache.get(video_id)
    except video_cache.VideoNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    target = entry.path.resolve()
    if not target.is_file() or not is_within(target, video_cache.VIDEO_CACHE_DIR):
        raise HTTPException(status_code=404, detail="Cached video file is gone")
    return range_stream_file(target, request, media_type=get_video_media_type(target))


@app.post("/api/v1/video/{video_id}/save")
def save_cached_video(video_id: str):
    """Keep it: moves the video into the media library."""
    from app import video as video_cache

    try:
        return video_cache.save(video_id)
    except video_cache.VideoNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except video_cache.UnsafeDelete as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/video/{video_id}")
def discard_cached_video(video_id: str):
    """The default when the viewer is closed. Only ever deletes Speedman's own copy."""
    from app import video as video_cache

    try:
        return video_cache.discard(video_id)
    except video_cache.VideoNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except video_cache.UnsafeDelete as e:
        logger.error(f"[video] refused unsafe delete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SyncTranscriptRequest(BaseModel):
    output_name: str = Field(..., description="Compressed filename to sync against")
    transcript_path: Optional[str] = Field(
        None,
        description="Media API's transcript JSON. Defaults to locating it automatically.",
    )
    source_path: Optional[str] = Field(None, description="Source audio, if not derivable")
    job_id: Optional[str] = Field(None, description="Optional Media API job ID to retrieve output path from")


@app.get("/api/v1/transcript/{output_name:path}")
def get_synced_transcript(
    output_name: str,
    format: Optional[str] = Query(None, description="Export format: srt, vtt, txt, or json"),
    source: bool = Query(False, description="Whether to use 1x source timeline"),
    timestamps: bool = Query(False, description="For txt format: include timestamps"),
):
    """The synced transcript for a compressed output, or 404 if it has not been made.

    Saves the UI guessing at '<stem>.synced.json' and keeps the naming in one place.
    Supports ?format=srt|vtt|txt for direct export download.
    """
    stem = safe_stem(output_name)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid output filename")
    target = (OUTPUT_DIR / f"{stem}.synced.json").resolve()
    if not target.is_file() or not is_within(target, OUTPUT_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"No synced transcript for '{stem}'. Transcribe the source, then sync.")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        segs = data.get("segments", [])

        # Ensure 1x source WebVTT exists on disk for video playback
        src_vtt = OUTPUT_DIR / f"{stem}.source.vtt"
        if not src_vtt.is_file() and segs:
            try:
                src_vtt.write_text(transcript_sync.to_source_vtt(segs), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[transcript] Could not auto-generate missing source.vtt: {e}")

        vtt = OUTPUT_DIR / f"{stem}.vtt"
        if not vtt.is_file() and segs:
            try:
                vtt.write_text(transcript_sync.to_vtt(segs), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[transcript] Could not auto-generate missing vtt: {e}")

        srt = OUTPUT_DIR / f"{stem}.srt"
        if not srt.is_file() and segs:
            try:
                srt.write_text(transcript_sync.to_srt(segs, use_source=False), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[transcript] Could not auto-generate missing srt: {e}")

        src_srt = OUTPUT_DIR / f"{stem}.source.srt"
        if not src_srt.is_file() and segs:
            try:
                src_srt.write_text(transcript_sync.to_srt(segs, use_source=True), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[transcript] Could not auto-generate missing source.srt: {e}")

        txt = OUTPUT_DIR / f"{stem}.txt"
        if not txt.is_file() and segs:
            try:
                txt.write_text(transcript_sync.to_plain_text(segs, include_timestamps=False), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[transcript] Could not auto-generate missing txt: {e}")

        # Formatted export download if requested
        if format:
            fmt = format.lower().strip()
            if fmt == "srt":
                content = transcript_sync.to_srt(segs, use_source=source)
                filename = f"{stem}{'.source' if source else ''}.srt"
                return Response(
                    content=content,
                    media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            elif fmt == "vtt":
                content = transcript_sync.to_source_vtt(segs) if source else transcript_sync.to_vtt(segs)
                filename = f"{stem}{'.source' if source else ''}.vtt"
                return Response(
                    content=content,
                    media_type="text/vtt",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            elif fmt == "txt":
                content = transcript_sync.to_plain_text(segs, include_timestamps=timestamps, use_source=source)
                filename = f"{stem}{'.source' if source else ''}.txt"
                return Response(
                    content=content,
                    media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            elif fmt != "json":
                raise HTTPException(status_code=400, detail=f"Unsupported format '{format}'. Use srt, vtt, txt, or json.")

        data["vtt_url"] = f"/api/v1/audio/{stem}.vtt"
        data["source_vtt_url"] = f"/api/v1/audio/{stem}.source.vtt"
        data["srt_url"] = f"/api/v1/audio/{stem}.srt"
        data["source_srt_url"] = f"/api/v1/audio/{stem}.source.srt"
        data["txt_url"] = f"/api/v1/audio/{stem}.txt"
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read transcript: {e}")


@app.get("/api/v1/transcribe/status/{job_id:path}")
def get_transcribe_job_status(job_id: str):
    """Proxies Media API's transcription job status so the UI can report real-time STT progress."""
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id is required")
    if not check_media_api_online():
        raise HTTPException(status_code=503, detail="Media API is currently offline")
    try:
        return media_api.get_media_api_job_status(job_id.strip())
    except RuntimeError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=502, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MUST stay above /api/v1/transcribe/{filename:path}. FastAPI matches in declaration
# order and a :path parameter is greedy, so a later /sync would be read as a filename --
# which it silently was, returning a plain transcription instead of a sync.
@app.post("/api/v1/transcribe/sync")
def sync_transcript_to_output(req: SyncTranscriptRequest):
    """Warp a 1x transcript onto a compressed output's timeline.

    Produces <stem>.vtt beside the audio, which players load as a subtitle track, plus
    <stem>.source.vtt for 1x video playback, and <stem>.synced.json with sample-accurate positions.
    Naive `t / N` drifts 0.08-0.23s because pauses compress harder than speech; the stored map is exact.
    """
    transcript_json: Optional[Path] = None

    if req.job_id:
        try:
            job_status = media_api.get_media_api_job_status(req.job_id)
            if job_status.get("status") == "completed":
                cand = job_status.get("output_files", {}).get("transcript_json")
                if cand:
                    p = normalize_path(cand).resolve()
                    if p.is_file() and is_path_allowed(p):
                        transcript_json = p
        except Exception as e:
            logger.debug(f"[sync] Media API job status lookup error: {e}")

    if not transcript_json and req.transcript_path:
        transcript_json = normalize_path(req.transcript_path).resolve()

    if not transcript_json:
        source = (normalize_path(req.source_path).resolve() if req.source_path
                  else find_source_for_output(req.output_name))
        if source is None:
            raise HTTPException(
                status_code=400,
                detail="Provide transcript_path or source_path -- the transcript could "
                       "not be located from the output name alone.")
        transcript_json = find_transcript_for_source(source, req.output_name)
        if transcript_json is None:
            transcript_json = source.with_suffix(".transcript.json")

    if not is_path_allowed(transcript_json):
        raise HTTPException(status_code=403, detail=f"'{transcript_json}' is outside permitted roots")
    if not transcript_json.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No transcript at {transcript_json}. Transcribe the source first, and "
                   "wait for the Media API job to finish before syncing.")

    try:
        result = transcript_sync.sync_to_output(transcript_json, req.output_name, OUTPUT_DIR)
    except timemap_store.TimeMapNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "synced",
        "output": req.output_name,
        "segments": result.segment_count,
        "source_duration_s": result.source_duration_s,
        "output_duration_s": result.output_duration_s,
        "vtt_path": str(result.vtt_path),
        "windows_vtt_path": to_windows_path(result.vtt_path),
        "source_vtt_path": str(result.source_vtt_path),
        "windows_source_vtt_path": to_windows_path(result.source_vtt_path),
        "srt_path": str(result.srt_path),
        "windows_srt_path": to_windows_path(result.srt_path),
        "source_srt_path": str(result.source_srt_path),
        "windows_source_srt_path": to_windows_path(result.source_srt_path),
        "txt_path": str(result.txt_path),
        "windows_txt_path": to_windows_path(result.txt_path),
        "json_path": str(result.json_path),
        "vtt_url": f"/api/v1/audio/{result.vtt_path.name}",
        "source_vtt_url": f"/api/v1/audio/{result.source_vtt_path.name}",
        "srt_url": f"/api/v1/audio/{result.srt_path.name}",
        "source_srt_url": f"/api/v1/audio/{result.source_srt_path.name}",
        "txt_url": f"/api/v1/audio/{result.txt_path.name}",
    }


class ImportTranscriptRequest(BaseModel):
    output_name: str
    content: str
    filename: Optional[str] = None


@app.post("/api/v1/transcript/import")
def import_transcript_for_output(req: ImportTranscriptRequest):
    """Warps an uploaded/provided subtitle string (VTT, SRT, or JSON) onto the output's timeline."""
    if not req.output_name:
        raise HTTPException(status_code=400, detail="output_name is required")
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="Transcript content cannot be empty")

    try:
        segments = transcript_sync.parse_subtitle_content(req.content, filename_hint=req.filename or "")
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=f"Subtitle parsing error: {e}")

    try:
        result = transcript_sync.sync_segments_to_output(segments, req.output_name, OUTPUT_DIR)
    except timemap_store.TimeMapNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "synced",
        "output": req.output_name,
        "segments": result.segment_count,
        "source_duration_s": result.source_duration_s,
        "output_duration_s": result.output_duration_s,
        "vtt_path": str(result.vtt_path),
        "windows_vtt_path": to_windows_path(result.vtt_path),
        "source_vtt_path": str(result.source_vtt_path),
        "windows_source_vtt_path": to_windows_path(result.source_vtt_path),
        "srt_path": str(result.srt_path),
        "windows_srt_path": to_windows_path(result.srt_path),
        "source_srt_path": str(result.source_srt_path),
        "windows_source_srt_path": to_windows_path(result.source_srt_path),
        "txt_path": str(result.txt_path),
        "windows_txt_path": to_windows_path(result.txt_path),
        "json_path": str(result.json_path),
        "vtt_url": f"/api/v1/audio/{result.vtt_path.name}",
        "source_vtt_url": f"/api/v1/audio/{result.source_vtt_path.name}",
        "srt_url": f"/api/v1/audio/{result.srt_path.name}",
        "source_srt_url": f"/api/v1/audio/{result.source_srt_path.name}",
        "txt_url": f"/api/v1/audio/{result.txt_path.name}",
    }


@app.post("/api/v1/transcribe/pipeline")
def transcribe_pipeline(req: TranscribePipelineRequest):
    """End-to-end headless pipeline: starts transcription, optionally waits for ASR completion,
    warps the transcript onto the compressed output timeline (.vtt, .srt, .txt, .synced.json),
    detects semantic chapters, and optionally exports to Obsidian.
    """
    stem = safe_stem(req.output_name)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid output filename")

    if req.source_path:
        source = normalize_path(req.source_path).resolve()
        if not is_path_allowed(source):
            raise HTTPException(
                status_code=403,
                detail=f"Source path '{source}' is outside permitted workstation roots",
            )
    else:
        source = find_source_for_output(req.output_name)
        if source is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not determine source audio for '{req.output_name}'. Provide source_path.",
            )

    if not source.is_file():
        raise HTTPException(status_code=404, detail=f"Source audio not found: {source}")

    if looks_like_speedman_output(source):
        raise HTTPException(
            status_code=400,
            detail=f"'{source.name}' is a Speedman output, not a source. Transcribing compressed audio returns nonsense; pass the original file.",
        )

    if not check_media_api_online():
        raise HTTPException(
            status_code=503,
            detail="Media API (127.0.0.1:8080) is currently offline. Ensure Media API is running.",
        )

    try:
        media_resp = send_to_media_api_transcribe(source, engine=req.engine)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    job_id = media_resp.get("job_id") if isinstance(media_resp, dict) else None
    if not job_id:
        raise HTTPException(status_code=502, detail="Media API did not return a valid job_id")

    if not req.wait:
        return {
            "status": "forwarded_to_media_api",
            "job_id": job_id,
            "output_name": req.output_name,
            "source_path": str(source),
            "windows_source_path": to_windows_path(source),
            "media_api_response": media_resp,
        }

    # If wait=True, poll until finished or timeout
    poll_interval = 0.5
    t_start = time.monotonic()
    last_status = None
    transcript_json: Optional[Path] = None

    while time.monotonic() - t_start < req.timeout_s:
        try:
            status_info = media_api.get_media_api_job_status(job_id)
            last_status = status_info.get("status")
            if last_status == "completed":
                cand = status_info.get("output_files", {}).get("transcript_json")
                if cand:
                    p = normalize_path(cand).resolve()
                    if p.is_file() and is_path_allowed(p):
                        transcript_json = p
                break
            elif last_status in ("failed", "error"):
                err_msg = status_info.get("error", "Unknown Media API transcription error")
                raise HTTPException(status_code=502, detail=f"Media API transcription failed: {err_msg}")
        except HTTPException:
            raise
        except Exception as e:
            logger.debug(f"[pipeline] Polling Media API job status error: {e}")

        time.sleep(poll_interval)

    if last_status != "completed":
        raise HTTPException(
            status_code=504,
            detail=f"Media API transcription job '{job_id}' timed out after {req.timeout_s}s",
        )

    # Sync to output
    if not transcript_json:
        transcript_json = find_transcript_for_source(source, req.output_name)
        if transcript_json is None:
            transcript_json = source.with_suffix(".transcript.json")

    if not transcript_json or not transcript_json.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Transcript file '{transcript_json}' not found after job completed",
        )

    try:
        sync_result = transcript_sync.sync_to_output(transcript_json, req.output_name, OUTPUT_DIR)
    except timemap_store.TimeMapNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sync_payload = {
        "status": "synced",
        "output": req.output_name,
        "segments": sync_result.segment_count,
        "source_duration_s": sync_result.source_duration_s,
        "output_duration_s": sync_result.output_duration_s,
        "vtt_path": str(sync_result.vtt_path),
        "windows_vtt_path": to_windows_path(sync_result.vtt_path),
        "txt_path": str(sync_result.txt_path),
        "windows_txt_path": to_windows_path(sync_result.txt_path),
        "json_path": str(sync_result.json_path),
    }

    # Chapters
    chapters_payload = None
    if req.auto_chapters:
        try:
            from app import chapters
            t_data = json.loads(sync_result.json_path.read_text(encoding="utf-8"))
            segs = t_data.get("segments", [])
            dur = float(t_data.get("output_duration_s") or sync_result.output_duration_s or 0.0)
            chaps = chapters.detect_and_save_chapters(
                output_name=req.output_name,
                segments=segs,
                duration_s=dur,
                output_dir=OUTPUT_DIR,
                use_llm=req.use_llm,
            )
            chapters_payload = {"count": len(chaps), "chapters": chaps}
        except Exception as e:
            logger.warning(f"[pipeline] Chapter generation failed: {e}")

    # Obsidian Export
    obsidian_payload = None
    if req.auto_obsidian:
        try:
            from app import obsidian
            t_data = json.loads(sync_result.json_path.read_text(encoding="utf-8"))
            segs = t_data.get("segments", [])
            dur = float(t_data.get("source_duration_s") or sync_result.source_duration_s or 0.0)
            speed = float(t_data.get("speed") or 1.0)
            summary_data = obsidian.generate_note_summary(segs, use_llm=req.use_llm)
            md_content = obsidian.format_obsidian_markdown(
                title=stem,
                source_name=source.name,
                duration_s=dur,
                speed=speed,
                summary_data=summary_data,
                segments=segs,
                include_transcript=True,
            )
            obsidian_payload = obsidian.save_obsidian_note(
                note_title=stem,
                content=md_content,
                folder=req.obsidian_folder,
            )
        except Exception as e:
            logger.warning(f"[pipeline] Obsidian export failed: {e}")

    return {
        "status": "success",
        "job_id": job_id,
        "output_name": req.output_name,
        "sync": sync_payload,
        "chapters": chapters_payload,
        "obsidian_note": obsidian_payload,
    }


# --------------------------------------------------------------------------- Obsidian Note Export

@app.get("/api/v1/export/obsidian/folders")
def get_obsidian_folders():
    from app import obsidian
    return {"folders": obsidian.get_available_folders(), "default": "Summaries"}


@app.post("/api/v1/export/obsidian")
def export_to_obsidian(req: ObsidianExportRequest):
    from app import obsidian
    stem = safe_stem(req.output_name)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid output filename")

    target_synced = (OUTPUT_DIR / f"{stem}.synced.json").resolve()
    target_src = (OUTPUT_DIR / f"{stem}.transcript.json").resolve()
    transcript_data = {}
    if target_synced.is_file() and is_within(target_synced, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_synced.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif target_src.is_file() and is_within(target_src, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_src.read_text(encoding="utf-8"))
        except Exception:
            pass

    segments = transcript_data.get("segments", [])
    duration_s = float(transcript_data.get("source_duration_s") or transcript_data.get("duration") or 0.0)
    speed = float(transcript_data.get("speed") or 1.0)
    source_name = transcript_data.get("source_path") or stem

    summary_data = None
    if req.include_summary:
        summary_data = obsidian.generate_note_summary(segments, use_llm=req.use_llm)

    note_title = req.note_title.strip() if req.note_title else stem
    audio_fname = req.output_name if req.output_name.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac")) else f"{stem}.wav"
    formatted_md = obsidian.format_obsidian_markdown(
        title=note_title,
        source_name=source_name,
        duration_s=duration_s,
        speed=speed,
        summary_data=summary_data,
        bookmarks=req.bookmarks if req.include_highlights else [],
        segments=segments if req.include_transcript else [],
        custom_tags=req.custom_tags,
        include_transcript=req.include_transcript,
        embed_audio=req.embed_audio,
        audio_filename=audio_fname,
        deep_link_timestamps=req.deep_link_timestamps,
    )

    try:
        return obsidian.save_obsidian_note(
            note_title=note_title,
            content=formatted_md,
            folder=req.folder,
        )
    except obsidian.ObsidianExportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[obsidian] Note save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write note: {e}")


@app.post("/api/v1/export/obsidian/daily-digest")
def export_quotes_to_daily_digest(req: DailyDigestExportRequest):
    from app import obsidian
    try:
        return obsidian.append_quotes_to_daily_note(
            quotes=req.quotes,
            source_title=req.source_title,
            folder=req.folder,
            date_str=req.date_str,
        )
    except obsidian.ObsidianExportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[obsidian] Daily digest append failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to append to daily digest: {e}")


@app.get("/api/v1/tools/cache-stats")
def get_cache_stats_route():
    from app import cache_cleaner
    try:
        return cache_cleaner.get_cache_stats()
    except Exception as e:
        logger.error(f"[cache] Failed to fetch cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate cache stats: {e}")


@app.post("/api/v1/tools/clean-cache")
def clean_cache_route(req: Optional[CleanCacheRequest] = None):
    from app import cache_cleaner
    req = req or CleanCacheRequest()
    try:
        return cache_cleaner.clean_cache(max_age_days=req.days, all_files=req.all_files)
    except Exception as e:
        logger.error(f"[cache] Failed to clean cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clean cache: {e}")


@app.post("/api/v1/export/obsidian/open")
def open_obsidian_note(req: OpenObsidianNoteRequest):
    from app import obsidian
    p = normalize_path(req.path)
    if not is_within(p, obsidian.NOTEBOOK_ROOT):
        raise HTTPException(status_code=403, detail="Note path is outside permitted 1Notebook directory")
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Note file not found: {p}")

    win_path = to_windows_path(p)
    try:
        # Prefer direct obsidian:// URI so Windows opens directly in Obsidian app
        try:
            rel = p.resolve().relative_to(obsidian.VAULT_ROOT.resolve())
            rel_str = str(rel.with_suffix("")).replace("\\", "/")
            obsidian_uri = f"obsidian://open?vault={obsidian.VAULT_NAME}&file={urllib.parse.quote(rel_str, safe='')}"
            subprocess.Popen(["cmd.exe", "/c", "start", "", obsidian_uri])
            return {"status": "success", "opened": win_path, "obsidian_uri": obsidian_uri}
        except Exception:
            subprocess.Popen(["cmd.exe", "/c", "start", "", win_path])
            return {"status": "success", "opened": win_path}
    except Exception as e:
        logger.error(f"[obsidian] Failed to open note in Windows: {e}")
        raise HTTPException(status_code=500, detail=f"Could not open note: {e}")


# --------------------------------------------------------------------------- Chapters & TOC

@app.get("/api/v1/chapters/{output_name}")
def get_chapters(output_name: str, use_llm: bool = Query(True)):
    """Fetch chapters for output_name, generating them if not already cached."""
    from app import chapters
    stem = safe_stem(output_name)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid output filename")

    cached = chapters.load_cached_chapters(output_name, OUTPUT_DIR)
    if cached is not None:
        return {"status": "cached", "stem": stem, "chapters": cached}

    target_synced = (OUTPUT_DIR / f"{stem}.synced.json").resolve()
    target_src = (OUTPUT_DIR / f"{stem}.transcript.json").resolve()
    transcript_data = {}
    if target_synced.is_file() and is_within(target_synced, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_synced.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif target_src.is_file() and is_within(target_src, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_src.read_text(encoding="utf-8"))
        except Exception:
            pass

    segments = transcript_data.get("segments", [])
    duration_s = float(transcript_data.get("output_duration_s") or transcript_data.get("duration") or 0.0)

    if not segments:
        return {"status": "empty", "stem": stem, "chapters": []}

    generated = chapters.detect_and_save_chapters(
        output_name=output_name,
        segments=segments,
        duration_s=duration_s,
        output_dir=OUTPUT_DIR,
        use_llm=use_llm,
    )
    return {"status": "generated", "stem": stem, "chapters": generated}


@app.post("/api/v1/chapters/{output_name}")
def generate_chapters_route(output_name: str, req: Optional[GenerateChaptersRequest] = None):
    """Force re-generate or compute chapters for output_name."""
    from app import chapters
    req = req or GenerateChaptersRequest()
    stem = safe_stem(output_name)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid output filename")

    if not req.force_regenerate:
        cached = chapters.load_cached_chapters(output_name, OUTPUT_DIR)
        if cached is not None:
            return {"status": "cached", "stem": stem, "chapters": cached}

    target_synced = (OUTPUT_DIR / f"{stem}.synced.json").resolve()
    target_src = (OUTPUT_DIR / f"{stem}.transcript.json").resolve()
    transcript_data = {}
    if target_synced.is_file() and is_within(target_synced, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_synced.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif target_src.is_file() and is_within(target_src, OUTPUT_DIR):
        try:
            transcript_data = json.loads(target_src.read_text(encoding="utf-8"))
        except Exception:
            pass

    segments = transcript_data.get("segments", [])
    duration_s = float(transcript_data.get("output_duration_s") or transcript_data.get("duration") or 0.0)
    if not segments:
        raise HTTPException(status_code=400, detail=f"No transcript found for '{stem}' to generate chapters")

    generated = chapters.detect_and_save_chapters(
        output_name=output_name,
        segments=segments,
        duration_s=duration_s,
        output_dir=OUTPUT_DIR,
        use_llm=req.use_llm,
    )
    return {"status": "generated", "stem": stem, "chapters": generated}


# --------------------------------------------------------------------------- In-Place Transcript Editing

@app.post("/api/v1/transcript/{output_name}/edit-cue")
def edit_transcript_cue_route(output_name: str, req: EditCueRequest):
    """Edit text for a single cue in-place, synchronizing .synced.json, .vtt, .srt, .txt."""
    try:
        res = transcript_sync.edit_transcript_cue(
            output_name=output_name,
            line_index=req.line_index,
            new_text=req.new_text,
            output_dir=OUTPUT_DIR,
        )
        return res
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[transcript] Cue edit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to edit cue: {e}")


@app.post("/api/v1/transcript/{output_name}/find-replace")
def find_replace_transcript_route(output_name: str, req: FindReplaceRequest):
    """Batch find-and-replace across transcript cues, synchronizing all subtitle files."""
    try:
        res = transcript_sync.batch_find_and_replace(
            output_name=output_name,
            find_text=req.find,
            replace_text=req.replace,
            match_case=req.match_case,
            output_dir=OUTPUT_DIR,
        )
        return res
    except transcript_sync.TranscriptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[transcript] Find and replace failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find and replace: {e}")


@app.get("/api/v1/waveform/{filename:path}")
def get_waveform_peaks(filename: str):
    """Returns normalized amplitude envelope peaks (120 bins) for waveform scrubbing."""
    stem = safe_stem(filename)
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid audio filename")
    peaks_file = (OUTPUT_DIR / f"{stem}.peaks.json").resolve()
    if is_within(peaks_file, OUTPUT_DIR) and peaks_file.is_file():
        try:
            return json.loads(peaks_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        target_audio = resolve_output_file(filename)
    except HTTPException:
        matches = list(OUTPUT_DIR.glob(f"{stem}.*"))
        audio_matches = [m for m in matches if m.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")]
        if audio_matches:
            target_audio = audio_matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found in output directory")

    try:
        from speedman.post import compute_waveform_peaks
        y = sio.load(target_audio)
        peaks = compute_waveform_peaks(y, num_bins=120)
        payload = {"peaks": peaks, "filename": target_audio.name}
        try:
            peaks_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute waveform peaks: {e}")


@app.post("/api/v1/transcribe/{filename:path}")
def transcribe_audio_with_media_api(filename: str, req: Optional[TranscribeRequest] = None):
    """Transcribes the ORIGINAL audio a compressed file was made from.

    This route used to send Speedman's own output to Media API, which produced garbage:
    measured on a 45s clip, parakeet returns 150 coherent words at 1x and 11 words of
    nonsense at 5x (WER 0.19 at 3x, 0.96 at 5x -- see docs/EVALUATION.md). ASR engines are
    not trained on time-compressed speech and collapse exactly where Speedman becomes
    useful, so transcription runs on the source and never on the result.
    """
    req = req or TranscribeRequest()

    if req.source_path:
        source = normalize_path(req.source_path).resolve()
        if not is_path_allowed(source):
            raise HTTPException(
                status_code=403,
                detail=f"Source path '{source}' is outside permitted workstation roots")
    else:
        source = find_source_for_output(filename)
        if source is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not work out what '{Path(filename).name}' was made from. "
                    "Pass 'source_path' (compression results include it) -- transcribing "
                    "the compressed audio instead would return nonsense."
                ),
            )

    if not source.is_file():
        raise HTTPException(status_code=404, detail=f"Source audio not found: {source}")

    # Refuse rather than silently produce a useless transcript.
    if looks_like_speedman_output(source):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{source.name}' is a Speedman output, not a source. Transcribing "
                "time-compressed audio returns nonsense; pass the original file instead."
            ),
        )

    if not check_media_api_online():
        raise HTTPException(
            status_code=503,
            detail=(
                "Media API (127.0.0.1:8080) is currently offline. Please ensure "
                "the Media API service is running on the workstation before requesting transcription."
            ),
        )

    try:
        res = send_to_media_api_transcribe(source, engine=req.engine)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    job_id = res.get("job_id") if isinstance(res, dict) else None
    payload = {
        "status": "forwarded_to_media_api",
        "job_id": job_id,
        "file": source.name,
        "source_path": str(source),
        "windows_source_path": to_windows_path(source),
        "transcribed": "source_audio_at_1x",
        "media_api_response": res,
    }

    if req.sync_to_output:
        # Media API transcribes asynchronously, so the transcript is not on disk yet.
        # The client polls /api/v1/transcribe/sync once its job reports completion.
        payload["sync"] = {
            "status": "pending",
            "output": req.sync_to_output,
            "hint": "POST /api/v1/transcribe/sync when the Media API job completes",
        }
    return payload


# --------------------------------------------------------------------------- Blind A/B Evaluation

@app.post("/api/v1/compare")
async def compare_audio(
    file: Optional[UploadFile] = File(None),
    input_path: Optional[str] = Form(None),
    speeds_str: str = Form("4.0,5.0,6.0"),
):
    jobs.inc_job()
    t_start = time.perf_counter()
    temp_in = None
    try:
        speeds = [float(s.strip()) for s in speeds_str.split(",") if s.strip()]
        if not speeds:
            speeds = [4.0, 5.0, 6.0]

        if file is not None:
            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                temp_in = Path(tf.name)
                shutil.copyfileobj(file.file, tf)
            src_path = temp_in
            stem = Path(file.filename or "compare").stem
        elif input_path:
            src_path = normalize_path(input_path).resolve()
            if not is_path_allowed(src_path):
                raise HTTPException(status_code=403, detail=f"Input path '{src_path}' is outside permitted workstation roots")
            if not src_path.is_file():
                raise HTTPException(status_code=404, detail=f"Input file not found: {input_path}")
            stem = src_path.stem
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'input_path'")

        folder_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}"
        dest_dir = COMPARE_DIR / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        y = sio.load(src_path, 24000)
        sio.save(dest_dir / "original.wav", y, 24000)

        variants = []
        for sp in speeds:
            variants.append((f"{sp:g}x_uniform", sp, "fast", True))
            for pr in ("natural", "fast", "aggressive"):
                variants.append((f"{sp:g}x_{pr}", sp, pr, False))

        random.seed(int(time.time()))
        shuffled = variants.copy()
        random.shuffle(shuffled)

        key_lines = []
        key_data = {}
        tracks = []
        for idx, (lbl, sp, pr, uni) in enumerate(shuffled, start=1):
            blind_id = f"track_{idx:02d}.wav"
            cfg = build_config(speed=sp, preset=pr, backend="rubberband", uniform=uni)
            res = process(y, 24000, cfg)
            sio.save(dest_dir / blind_id, res.audio, res.sr)
            desc = f"{lbl} (speed={sp}x preset={pr} uniform={uni})"
            key_lines.append(f"{blind_id} -> {desc}")
            key_data[blind_id] = {
                "label": lbl,
                "speed": sp,
                "preset": pr,
                "uniform": uni,
                "duration_s": round(len(res.audio) / res.sr, 2),
                "effective_speech_rate": res.notes.get("effective_speech_rate", sp),
            }
            tracks.append({
                "track_id": blind_id,
                "url": f"/api/v1/audio/comparisons/{folder_name}/{blind_id}",
                "duration_s": round(len(res.audio) / res.sr, 2),
            })

        (dest_dir / "KEY.txt").write_text("\n".join(key_lines) + "\n", encoding="utf-8")
        (dest_dir / "key.json").write_text(json.dumps(key_data, indent=2), encoding="utf-8")

        elapsed = time.perf_counter() - t_start
        return {
            "status": "success",
            "folder_id": folder_name,
            "folder_path": str(dest_dir),
            "windows_folder_path": to_windows_path(dest_dir),
            "original_url": f"/api/v1/audio/comparisons/{folder_name}/original.wav",
            "original_duration_s": round(len(y) / 24000, 2),
            "tracks": tracks,
            "total_variants": len(tracks),
            "processing_time_s": round(elapsed, 2),
        }
    finally:
        if temp_in and temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass
        jobs.dec_job()


@app.get("/api/v1/comparisons")
def list_comparisons():
    if not COMPARE_DIR.exists():
        return []
    folders = sorted([d for d in COMPARE_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    results = []
    for d in folders[:20]:
        has_key = (d / "key.json").is_file()
        tracks = [p.name for p in sorted(d.glob("track_*.wav"))]
        results.append({
            "folder_id": d.name,
            "created": time.ctime(d.stat().st_mtime),
            "track_count": len(tracks),
            "has_key": has_key,
        })
    return results


@app.get("/api/v1/comparisons/{folder_id}/key")
def get_comparison_key(folder_id: str):
    clean_id = Path(folder_id).name
    dest_dir = COMPARE_DIR / clean_id
    if not dest_dir.is_dir() or not is_path_allowed(dest_dir):
        raise HTTPException(status_code=404, detail="Comparison folder not found")

    json_key = dest_dir / "key.json"
    if json_key.is_file():
        return json.loads(json_key.read_text(encoding="utf-8"))

    txt_key = dest_dir / "KEY.txt"
    if txt_key.is_file():
        return {"raw": txt_key.read_text(encoding="utf-8")}

    raise HTTPException(status_code=404, detail="Key mapping not found for this comparison")


# --------------------------------------------------------------------------- Listening Test
#
# Forced-choice A/B against a uniform control, run as a randomised speed ladder. This is
# the only instrument that reaches 5-6x: ASR saturates above 3.5x and the modulation
# metric structurally favours uniform. See docs/EVALUATION.md.
#
# Blinding is enforced here, not in the client: slot -> condition never appears in a
# payload, URL or filename until the session is complete.

@app.post("/api/v1/listening/session")
def create_listening_session(req: ListeningSessionRequest):
    """Renders every trial up front and returns the blind trial list."""
    from eval import listening, manifest as eval_manifest

    try:
        clips = eval_manifest.by_label(req.clips) if req.clips else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    jobs.inc_job()
    try:
        session = listening.build_session(
            speeds=req.speeds,
            trials_per_speed=req.trials_per_speed,
            clips=clips,
            excerpt_s=req.excerpt_s,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[listening] session build failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not build session: {e}")
    finally:
        jobs.dec_job()

    logger.info(f"[listening] session {session.session_id}: {len(session.trials)} trials")
    return session.blind_state()


@app.get("/api/v1/listening/clips")
def list_listening_clips():
    """The eval clip set, so listening verdicts sit alongside the WER numbers for the
    same audio."""
    from eval import manifest as eval_manifest

    return [
        {
            "label": c.label,
            "kind": c.kind,
            "duration_s": c.duration_s,
            "available": c.available,
        }
        for c in eval_manifest.CLIPS
    ]


@app.get("/api/v1/listening/sessions")
def list_listening_sessions():
    from eval import listening

    return listening.list_sessions()


@app.get("/api/v1/listening/session/{session_id}")
def get_listening_session(session_id: str):
    """Resume a session. Verdicts survive a browser crash."""
    from eval import listening

    try:
        return listening.load_session(session_id).blind_state()
    except listening.SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/listening/session/{session_id}/verdict")
def record_listening_verdict(session_id: str, req: ListeningVerdictRequest):
    from eval import listening

    try:
        session = listening.load_session(session_id)
        listening.record_verdict(session, req.trial_index, req.choice, req.followed)
    except listening.SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except listening.InvalidVerdict as e:
        raise HTTPException(status_code=400, detail=str(e))
    return session.blind_state()


@app.get("/api/v1/listening/session/{session_id}/results")
def get_listening_results(session_id: str):
    """Unblind. Refuses while trials remain, so a mid-session peek cannot bias the rest."""
    from eval import listening

    try:
        return listening.reveal(listening.load_session(session_id))
    except listening.SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except listening.InvalidVerdict as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/v1/listening/{session_id}/audio/{trial_index}/{slot}")
def stream_listening_audio(session_id: str, trial_index: int, slot: str, request: Request):
    """Audio addressed by slot, never by condition -- the URL itself must not unblind."""
    from eval import listening

    if slot not in ("a", "b"):
        raise HTTPException(status_code=400, detail="slot must be 'a' or 'b'")
    try:
        session = listening.load_session(session_id)
    except listening.SessionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    target = session.audio_path(trial_index, slot).resolve()
    if not target.is_file() or not is_within(target, listening.LISTENING_DIR):
        raise HTTPException(status_code=404, detail="Trial audio not found")
    return range_stream_file(target, request, media_type="audio/wav")


# --------------------------------------------------------------------------- Web UI Route

@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Speedman Speech Engine</h1><p>API is active, but web UI index.html was not found.</p>"
