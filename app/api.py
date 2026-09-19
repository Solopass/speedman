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
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from speedman import io as sio
from speedman.config import PRESETS, build_config
from speedman.pipeline import process
from app.range_response import range_stream_file
from app.queue import job_queue, SpeedmanJob
from app.media_api import (
    check_media_api_online,
    get_media_api_health,
    list_media_library,
    extract_audio_from_url,
    send_to_media_api_transcribe,
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


def normalize_path(path_input: str | Path) -> Path:
    r"""Seamlessly normalize Windows (D:\...) and WSL (/mnt/d/...) file paths."""
    s = str(path_input).strip().strip('"').strip("'")
    if re.match(r"^[a-zA-Z]:[\\/]", s):
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def to_windows_path(path_input: str | Path) -> str:
    r"""Convert WSL (/mnt/d/...) path to Windows (D:\...) path."""
    s = str(path_input).strip().strip('"').strip("'")
    if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
        drive = s[5].upper()
        rest = s[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return s


def is_path_allowed(target: Path) -> bool:
    """True when target resolves to one of ALLOWED_ROOTS or inside it."""
    try:
        resolved = target.resolve()
        for root in ALLOWED_ROOTS:
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved.is_relative_to(resolved_root):
                return True
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------- Session & Watchdog

class SessionManager:
    """Tracks active browser tabs/clients and active audio processing jobs."""
    def __init__(self):
        self.active_jobs: int = 0
        self.sessions: dict[str, float] = {}
        self.last_active_time: float = time.time()
        self.auto_close: bool = os.getenv("AUTO_CLOSE", "true").lower() in ("true", "1", "yes")
        self.idle_timeout: int = int(os.getenv("IDLE_TIMEOUT", "60"))

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
        self.sessions = {sid: ts for sid, ts in self.sessions.items() if now - ts < 12.0}
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
    watchdog = asyncio.create_task(idle_watchdog())
    yield
    watchdog.cancel()
    try:
        await watchdog
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    preset: str = Field("fast", description="Preset: natural | fast | aggressive | max")
    uniform: bool = Field(False, description="Uniform constant-rate stretch (control mode)")
    format: str = Field("wav", description="Audio output format: wav | mp3 | m4a | flac")
    output_filename: Optional[str] = None


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="YouTube or web media URL to extract and compress")
    speed: float = Field(5.0, ge=1.01, le=30.0)
    preset: str = Field("fast")
    uniform: bool = Field(False)
    format: str = Field("wav", description="Audio output format: wav | mp3 | m4a | flac")


class TranscribeRequest(BaseModel):
    engine: str = Field("parakeet", description="Speech-to-text engine: parakeet or whisper")


class CompareRequest(BaseModel):
    input_path: str = Field(..., description="Local path to audio file on workstation")
    speeds: List[float] = Field(default_factory=lambda: [4.0, 5.0, 6.0], description="List of target speeds to compare")


class HeartbeatRequest(BaseModel):
    session_id: str


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


def get_audio_media_type(path: Path) -> str:
    ext = path.suffix.lower()
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
    clean = filename.strip("/\\")
    target = (OUTPUT_DIR / clean).resolve()
    if target.is_file() and is_path_allowed(target):
        return range_stream_file(target, request, media_type=get_audio_media_type(target))

    file_name_only = Path(clean).name
    candidates = [p for p in OUTPUT_DIR.rglob(file_name_only) if p.is_file() and is_path_allowed(p)]
    if candidates:
        return range_stream_file(candidates[0], request, media_type=get_audio_media_type(candidates[0]))

    raise HTTPException(status_code=404, detail=f"Audio file '{clean}' not found")


# --------------------------------------------------------------------------- Synchronous Compression

def _run_compression(src_path: Path, stem: str, speed: float, preset: str, uniform: bool, format: str = "wav") -> dict[str, Any]:
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    if not is_path_allowed(src_path):
        raise HTTPException(status_code=403, detail=f"Input path '{src_path}' is outside permitted workstation roots")

    if not src_path.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {src_path}")

    t_start = time.perf_counter()
    cfg = build_config(speed=speed, preset=preset, backend="rubberband", uniform=uniform)
    y = sio.load(src_path, cfg.sample_rate)
    in_dur = len(y) / cfg.sample_rate

    res = process(y, cfg.sample_rate, cfg)

    fmt = format.lower().lstrip(".")
    ext = f".{fmt}" if fmt in ("wav", "mp3", "m4a", "flac") else ".wav"
    out_name = f"{stem}_{speed:g}x_{preset}{'_uniform' if uniform else ''}{ext}"
    out_path = OUTPUT_DIR / out_name
    sio.save(out_path, res.audio, res.sr)
    out_dur = len(res.audio) / res.sr

    elapsed = time.perf_counter() - t_start

    return {
        "status": "success",
        "filename": out_name,
        "output_path": str(out_path),
        "windows_output_path": to_windows_path(out_path),
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
        "processing_time_s": round(elapsed, 2),
        "timings": res.timings,
        "chunked": res.notes.get("chunked", False),
    }


@app.post("/api/v1/compress")
async def compress_audio_multipart(
    file: Optional[UploadFile] = File(None),
    input_path: Optional[str] = Form(None),
    speed: float = Form(5.0),
    preset: str = Form("fast"),
    uniform: bool = Form(False),
    format: str = Form("wav"),
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

        return _run_compression(src_path, stem, speed, preset, uniform, format=format)
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
        return _run_compression(src_path, stem, req.speed, req.preset, req.uniform, format=req.format)
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

    job = job_queue.submit(
        input_path=src_path,
        speed=req.speed,
        preset=req.preset,
        uniform=req.uniform,
        output_dir=OUTPUT_DIR,
        output_format=req.format,
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


@app.post("/api/v1/ingest/url")
async def ingest_url_and_compress(req: IngestUrlRequest):
    """Extracts audio from a YouTube/web URL and queues it for Speedman compression."""
    try:
        downloaded_path = extract_audio_from_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract audio from URL: {e}")

    job = job_queue.submit(
        input_path=downloaded_path,
        speed=req.speed,
        preset=req.preset,
        uniform=req.uniform,
        output_dir=OUTPUT_DIR,
        output_format=req.format,
    )
    return {
        "status": "queued",
        "job_id": job.job_id,
        "input_filename": downloaded_path.name,
        "speed": req.speed,
        "preset": req.preset,
        "format": job.output_format,
    }


@app.post("/api/v1/transcribe/{filename:path}")
def transcribe_audio_with_media_api(filename: str, req: Optional[TranscribeRequest] = None):
    """Sends an audio file from Speedman's output to Media API for Speech-To-Text."""
    req = req or TranscribeRequest()
    clean = filename.strip("/\\")
    target = (OUTPUT_DIR / clean).resolve()
    if not target.is_file() or not is_path_allowed(target):
        # Search by filename
        candidates = [p for p in OUTPUT_DIR.rglob(Path(clean).name) if p.is_file() and is_path_allowed(p)]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Audio file '{clean}' not found")
        target = candidates[0]

    try:
        res = send_to_media_api_transcribe(target, engine=req.engine)
        return {
            "status": "forwarded_to_media_api",
            "file": target.name,
            "media_api_response": res,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


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


# --------------------------------------------------------------------------- Web UI Route

@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Speedman Speech Engine</h1><p>API is active, but web UI index.html was not found.</p>"
