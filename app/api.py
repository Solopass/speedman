"""FastAPI Application for Speedman Speech Engine.

Supports on-demand systemd socket activation, 0-RAM standby, auto-close watchdog,
speech compression (1.01x to 30x), blind A/B comparison sets, and interactive Web UI.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from speedman import io as sio
from speedman.config import PRESETS, build_config
from speedman.pipeline import process
from app.range_response import range_stream_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [speedman] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("speedman_api")

OUTPUT_DIR = Path("/mnt/d/Audio/Speed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR = OUTPUT_DIR / "comparisons"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

class SessionManager:
    """Tracks active browser tabs/clients and active audio processing jobs.
    
    GUARANTEE: The server NEVER shuts down while a browser tab/window is open,
    and NEVER shuts down while audio jobs are running.
    The 60-second idle countdown starts ONLY AFTER you exit out of the browser app!
    """
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
        # Active if heartbeat received within last 12 seconds
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
        # NEVER shut down while jobs are running
        if self.active_jobs > 0:
            self.last_active_time = time.time()
            return False
        # NEVER shut down while browser tab is open
        if self.active_client_count() > 0:
            self.last_active_time = time.time()
            return False
        # Browser is closed and 0 active jobs: count down 60s
        idle = time.time() - self.last_active_time
        return idle >= self.idle_timeout

jobs = SessionManager()

async def idle_watchdog():
    """Monitors client sessions and active jobs. Shuts down 60s AFTER browser is closed."""
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

app = FastAPI(
    title="Speedman Speech Engine API",
    description="Speech-aware dynamic time compression microservice",
    version="0.1.0",
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

class CompressRequest(BaseModel):
    input_path: str = Field(..., description="Local path to audio file on workstation")
    speed: float = Field(5.0, ge=1.01, le=30.0, description="Speed multiplier (e.g. 5.0 for 5x)")
    preset: str = Field("fast", description="Preset: natural | fast | aggressive | max")
    uniform: bool = Field(False, description="Uniform constant-rate stretch (control mode)")
    output_filename: Optional[str] = None

class HeartbeatRequest(BaseModel):
    session_id: str

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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "speedman",
        "version": "0.1.0",
        "active_jobs": jobs.active_jobs,
        "active_clients": jobs.active_client_count(),
        "idle_seconds": int(time.time() - jobs.last_active_time),
        "output_dir": str(OUTPUT_DIR),
    }

@app.post("/api/v1/shutdown")
async def shutdown():
    import threading
    logger.info("[api] Shutdown requested via POST /api/v1/shutdown")
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return {"status": "shutting_down"}

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

@app.get("/api/v1/audio/{filename:path}")
def stream_audio(filename: str, request: Request):
    clean = Path(filename).name
    # Search in OUTPUT_DIR and nested comparisons
    candidates = list(OUTPUT_DIR.rglob(clean))
    if not candidates:
        raise HTTPException(status_code=404, detail="Audio file not found")
    target = candidates[0]
    return range_stream_file(target, request, media_type="audio/wav")

@app.post("/api/v1/compress")
async def compress_audio(
    file: Optional[UploadFile] = File(None),
    input_path: Optional[str] = Form(None),
    speed: float = Form(5.0),
    preset: str = Form("fast"),
    uniform: bool = Form(False),
):
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    jobs.inc_job()
    t_start = time.perf_counter()
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
            src_path = Path(input_path).expanduser().resolve()
            if not src_path.is_file():
                raise HTTPException(status_code=404, detail=f"Input file not found: {input_path}")
            stem = src_path.stem
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'input_path'")

        cfg = build_config(speed=speed, preset=preset, backend="rubberband", uniform=uniform)
        y = sio.load(src_path, cfg.sample_rate)
        in_dur = len(y) / cfg.sample_rate

        res = process(y, cfg.sample_rate, cfg)

        out_name = f"{stem}_{speed:g}x_{preset}{'_uniform' if uniform else ''}.wav"
        out_path = OUTPUT_DIR / out_name
        sio.save(out_path, res.audio, res.sr)
        out_dur = len(res.audio) / res.sr

        elapsed = time.perf_counter() - t_start

        return {
            "status": "success",
            "filename": out_name,
            "audio_url": f"/api/v1/audio/{out_name}",
            "speed": speed,
            "preset": preset,
            "uniform": uniform,
            "input_duration_s": round(in_dur, 2),
            "output_duration_s": round(out_dur, 2),
            "compression_ratio": round(in_dur / max(out_dur, 0.001), 2),
            "silence_fraction": res.notes.get("silence_fraction", 0.0),
            "effective_speech_rate": res.notes.get("effective_speech_rate", speed),
            "processing_time_s": round(elapsed, 2),
            "timings": res.timings,
        }
    finally:
        if temp_in and temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass
        jobs.dec_job()

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
            src_path = Path(input_path).expanduser().resolve()
            if not src_path.is_file():
                raise HTTPException(status_code=404, detail=f"Input file not found: {input_path}")
            stem = src_path.stem
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'input_path'")

        folder_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}"
        dest_dir = COMPARE_DIR / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Load reference audio
        y = sio.load(src_path, 24000)
        sio.save(dest_dir / "original.wav", y, 24000)

        # Build variants
        variants = []
        for sp in speeds:
            variants.append((f"{sp:g}x_uniform", sp, "fast", True))
            for pr in ("natural", "fast", "aggressive"):
                variants.append((f"{sp:g}x_{pr}", sp, pr, False))

        # Randomize for blind testing
        random.seed(int(time.time()))
        shuffled = variants.copy()
        random.shuffle(shuffled)

        key_lines = []
        tracks = []
        for idx, (lbl, sp, pr, uni) in enumerate(shuffled, start=1):
            blind_id = f"track_{idx:02d}.wav"
            cfg = build_config(speed=sp, preset=pr, backend="rubberband", uniform=uni)
            res = process(y, 24000, cfg)
            sio.save(dest_dir / blind_id, res.audio, res.sr)
            key_lines.append(f"{blind_id} -> {lbl} (speed={sp}x preset={pr} uniform={uni})")
            tracks.append({
                "track_id": blind_id,
                "url": f"/api/v1/audio/{blind_id}",
                "label": lbl,
            })

        (dest_dir / "KEY.txt").write_text("\n".join(key_lines) + "\n")

        elapsed = time.perf_counter() - t_start
        return {
            "status": "success",
            "folder": str(dest_dir),
            "original_url": f"/api/v1/audio/original.wav",
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

@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Speedman Speech Engine</h1><p>API is active. Web UI index.html not found.</p>"
