"""Asynchronous Job Queue and Process Cancellation for Speedman.

Manages background compression jobs on workstation E-cores, tracks measured progress %,
chunk stages, and ETAs, and supports clean cancellation.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from speedman import io as sio
from speedman.config import build_config
from speedman.pipeline import process
from speedman.chunking import CancelledError

from app.paths import to_windows_path

logger = logging.getLogger("speedman_queue")


@dataclass
class JobProgress:
    stage: str = "queued"
    progress_pct: float = 0.0
    current_chunk: int = 0
    total_chunks: int = 1
    elapsed_s: float = 0.0
    eta_s: Optional[float] = None


class SpeedmanJob:
    def __init__(
        self,
        job_id: str,
        input_path: Optional[Path] = None,
        speed: float = 5.0,
        preset: str = "fast",
        uniform: bool = False,
        output_dir: Optional[Path] = None,
        output_format: str = "wav",
        source_url: Optional[str] = None,
        include_video: bool = False,
    ):
        if input_path is None and not source_url:
            raise ValueError("a job needs either an input_path or a source_url")
        self.job_id = job_id
        self.include_video = include_video
        self.video_id: Optional[str] = None
        # Resolved by the worker when the job starts from a URL: downloading a two-hour
        # podcast inside the HTTP request would hold the connection open for minutes.
        self.source_url = source_url
        self.input_path = input_path
        self.speed = speed
        self.preset = preset
        self.uniform = uniform
        self.output_dir = output_dir or Path("/mnt/d/Audio/Speed")
        self.output_format = output_format.lower().lstrip(".")

        self.status = "queued"  # queued | processing | completed | failed | cancelled
        self.progress = JobProgress()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()
        if self.status in ("queued", "processing"):
            self.status = "cancelled"
            self.progress.stage = "cancelled"

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.progress.stage,
            "progress_pct": self.progress.progress_pct,
            "current_chunk": self.progress.current_chunk,
            "total_chunks": self.progress.total_chunks,
            "elapsed_s": round(self.progress.elapsed_s, 1),
            "eta_s": self.progress.eta_s,
            "speed": self.speed,
            "preset": self.preset,
            "uniform": self.uniform,
            "output_format": self.output_format,
            "input_filename": self.input_path.name if self.input_path else None,
            "source_url": self.source_url,
            "video_id": self.video_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    def __init__(self, max_history: int = 50):
        self.jobs: Dict[str, SpeedmanJob] = {}
        self.job_queue: queue.Queue[SpeedmanJob] = queue.Queue()
        self.max_history = max_history
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="speedman_worker")
        self._worker_thread.start()

    def submit(
        self,
        input_path: Optional[Path] = None,
        speed: float = 5.0,
        preset: str = "fast",
        uniform: bool = False,
        output_dir: Optional[Path] = None,
        output_format: str = "wav",
        source_url: Optional[str] = None,
        include_video: bool = False,
    ) -> SpeedmanJob:
        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = SpeedmanJob(
            job_id=job_id,
            input_path=input_path,
            speed=speed,
            preset=preset,
            uniform=uniform,
            output_dir=output_dir,
            output_format=output_format,
            source_url=source_url,
            include_video=include_video,
        )
        with self._lock:
            self.jobs[job_id] = job
            self._trim_history()

        self.job_queue.put(job)
        what = input_path.name if input_path else source_url
        logger.info(f"[queue] Job {job_id} queued for {what} ({speed}x {preset}, format={job.output_format})")
        return job

    def get_job(self, job_id: str) -> Optional[SpeedmanJob]:
        with self._lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            job.cancel()
            logger.info(f"[queue] Job {job_id} cancellation signaled")
            return True

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in sorted_jobs[:limit]]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self.jobs.values() if j.status in ("queued", "processing"))

    def _trim_history(self):
        if len(self.jobs) > self.max_history:
            terminal = [k for k, j in self.jobs.items() if j.status in ("completed", "failed", "cancelled")]
            terminal.sort(key=lambda k: self.jobs[k].created_at)
            for k in terminal[: len(self.jobs) - self.max_history]:
                self.jobs.pop(k, None)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job.is_cancelled():
                job.status = "cancelled"
                self.job_queue.task_done()
                continue

            self._execute_job(job)
            self.job_queue.task_done()

    @staticmethod
    def _stage_video_for(job: SpeedmanJob, on_prog) -> Path:
        """Download the video, move it into Speedman's cache, and register it.

        The move matters: discard() only ever deletes inside the cache, so keeping temp
        videos out of D:\\Output\\Videos means a bug here cannot reach the media library.
        A file that predated our request is copied instead of moved -- it belongs to the
        user and must stay where they left it.
        """
        import shutil as _shutil

        from app import video as video_cache
        from app.media_api import download_video_from_url

        downloaded, pre_existed = download_video_from_url(job.source_url, on_progress=on_prog)

        video_cache.VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target = video_cache.VIDEO_CACHE_DIR / downloaded.name
        if pre_existed:
            _shutil.copy2(str(downloaded), str(target))
        else:
            _shutil.move(str(downloaded), str(target))

        entry = video_cache.register(target, source_url=job.source_url, pre_existed=pre_existed)
        job.video_id = entry.video_id
        return target

    def _execute_job(self, job: SpeedmanJob):
        job.status = "processing"
        job.started_at = time.time()
        job.progress.stage = "loading"
        logger.info(f"[worker] Starting job {job.job_id} on {job.input_path}")

        t_start = time.perf_counter()

        def on_prog(data: str | dict):
            if isinstance(data, str):
                job.progress.stage = data
            elif isinstance(data, dict):
                job.progress.stage = data.get("stage", job.progress.stage)
                job.progress.progress_pct = data.get("progress_pct", job.progress.progress_pct)
                job.progress.current_chunk = data.get("current_chunk", job.progress.current_chunk)
                job.progress.total_chunks = data.get("total_chunks", job.progress.total_chunks)
                job.progress.elapsed_s = data.get("elapsed_s", job.progress.elapsed_s)
                job.progress.eta_s = data.get("eta_s", job.progress.eta_s)

        try:
            if job.source_url and job.input_path is None:
                job.progress.stage = "downloading"
                if job.include_video:
                    # One download, two uses: sio.load decodes audio straight out of the
                    # mp4 via ffmpeg, so asking for video does not cost a second fetch.
                    job.input_path = self._stage_video_for(job, on_prog)
                else:
                    from app.media_api import extract_audio_from_url

                    job.input_path = extract_audio_from_url(job.source_url, on_progress=on_prog)
                logger.info(f"[worker] Job {job.job_id} downloaded {job.input_path.name}")
                if job.is_cancelled():
                    raise CancelledError("Processing cancelled by user")
                job.progress.stage = "loading"
                job.progress.progress_pct = 0.0

            cfg = build_config(speed=job.speed, preset=job.preset, backend="rubberband", uniform=job.uniform)
            y = sio.load(job.input_path, cfg.sample_rate)
            in_dur = len(y) / cfg.sample_rate

            if job.is_cancelled():
                raise CancelledError("Processing cancelled by user")

            res = process(
                y,
                cfg.sample_rate,
                cfg,
                on_progress=on_prog,
                cancel_check=job.is_cancelled,
            )

            # A cancel landing after the last in-pipeline check would otherwise be
            # overwritten by "completed" below -- and write an output file nobody wants.
            if job.is_cancelled():
                raise CancelledError("Processing cancelled by user")

            stem = job.input_path.stem
            fmt = sio.normalize_output_format(job.output_format)
            out_name = f"{stem}_{job.speed:g}x_{job.preset}{'_uniform' if job.uniform else ''}.{fmt}"
            out_path = job.output_dir / out_name
            sio.save(out_path, res.audio, res.sr)
            out_dur = len(res.audio) / res.sr

            from app import timemap_store

            # Keep the map so a 1x transcript can be synced to this output later.
            timemap_store.save_quietly(out_name, res.time_map, job.speed, res.sr)

            elapsed = time.perf_counter() - t_start

            job.result = {
                "status": "success",
                "filename": out_name,
                "output_path": str(out_path),
                # The Studio shows this verbatim, so it has to match what /compress
                # returns -- otherwise queued jobs display a /mnt/d/... path.
                "windows_output_path": to_windows_path(out_path),
                # Provenance: transcription must run on this, never on the output above.
                "source_path": str(job.input_path),
                "windows_source_path": to_windows_path(job.input_path),
                "audio_url": f"/api/v1/audio/{out_name}",
                "speed": job.speed,
                "preset": job.preset,
                "uniform": job.uniform,
                "format": fmt,
                "input_duration_s": round(in_dur, 2),
                "output_duration_s": round(out_dur, 2),
                "compression_ratio": round(in_dur / max(out_dur, 0.001), 2),
                "silence_fraction": res.notes.get("silence_fraction", 0.0),
                "effective_speech_rate": res.notes.get("effective_speech_rate", job.speed),
                "estimated_speech_rate": res.notes.get("estimated_speech_rate", job.speed),
                # >0 means the map is running at its ceiling and has stopped being non-uniform.
                "rate_clamped_fraction": res.notes.get("rate_clamped_fraction", 0.0),
                "processing_time_s": round(elapsed, 2),
                "timings": res.timings,
                "chunked": res.notes.get("chunked", False),
                "video_id": job.video_id,
            }
            job.status = "completed"
            job.progress.stage = "completed"
            job.progress.progress_pct = 100.0
            job.progress.eta_s = 0.0
            job.finished_at = time.time()
            logger.info(f"[worker] Job {job.job_id} completed successfully in {elapsed:.1f}s")
        except CancelledError:
            job.status = "cancelled"
            job.progress.stage = "cancelled"
            job.finished_at = time.time()
            logger.info(f"[worker] Job {job.job_id} successfully cancelled")
        except Exception as e:
            job.status = "failed"
            job.progress.stage = "failed"
            job.error = str(e)
            job.finished_at = time.time()
            logger.error(f"[worker] Job {job.job_id} failed: {e}", exc_info=True)


job_queue = JobManager()
