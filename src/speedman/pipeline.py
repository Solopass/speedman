"""Orchestration: audio in, sped-up audio out."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import analyze, post, ratemap
from .config import Config
from .stretch import get_backend


@dataclass
class Result:
    audio: np.ndarray
    sr: int
    timings: dict
    notes: dict


CHUNK_THRESHOLD_MINUTES = 30.0


def process(
    y: np.ndarray,
    sr: int,
    cfg: Config,
    verbose: bool = False,
    on_progress: Optional[Callable[[str | dict], None]] = None,
    annotation=None,
    cancel_check: Optional[Callable[[], bool]] = None,
    chunk_target_min: float = 25.0,
) -> Result:
    minutes = len(y) / sr / 60.0

    # For files longer than 30 minutes, process in sequential pause-aligned chunks
    # to clamp peak memory and allow arbitrary audiobook/podcast length.
    if minutes > CHUNK_THRESHOLD_MINUTES and annotation is None:
        from .chunking import process_chunked
        out, t, notes = process_chunked(
            y, sr, cfg,
            target_chunk_min=chunk_target_min,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )
        return Result(audio=out, sr=sr, timings=t, notes=notes)

    t = {}
    notes: dict = {}
    backend = get_backend(cfg.backend)

    def step(msg):
        if on_progress:
            on_progress(msg)

    if cancel_check and cancel_check():
        from .chunking import CancelledError
        raise CancelledError("Processing cancelled by user")

    step("analysing")
    t0 = time.perf_counter()
    if cfg.uniform:
        tm = None
        notes["mode"] = "uniform"
    else:
        if not backend.supports_time_map:
            raise ValueError(
                f"backend {cfg.backend!r} cannot apply a time map, so it cannot do "
                "non-uniform compression. Use --backend rubberband, or pass --uniform "
                "to run this backend as a constant-rate control."
            )
        ann = annotation or analyze.annotate_vad(
            y, sr,
            protect_window_ms=cfg.ratemap.protect_window_ms,
            protect_lead_ms=cfg.ratemap.protect_lead_ms,
        )
        tm = ratemap.build_time_map(ann, cfg.speed, cfg.ratemap)
        notes.update(tm.notes)
        notes["mode"] = "non-uniform"
        notes["n_anchors"] = len(tm.anchors)
        s = tm.notes.get("silence_fraction", 0.0)
        # Measured from the solved map, not estimated. The old formula ignored the preset
        # and ran optimistic by up to 4.2% (docs/EVALUATION.md); it is kept alongside so
        # the eval harness can still compare the two.
        notes["effective_speech_rate"] = round(
            tm.measured_speech_rate([sp.kind.value for sp in ann.spans]), 2)
        notes["estimated_speech_rate"] = round(cfg.speed * (1.0 - 0.6 * s), 2)
        notes["rate_clamped_fraction"] = round(tm.clamped_fraction(cfg.ratemap.max_rate), 4)
    t["analyze"] = time.perf_counter() - t0

    if cancel_check and cancel_check():
        from .chunking import CancelledError
        raise CancelledError("Processing cancelled by user")

    step("stretching")
    t0 = time.perf_counter()
    if tm is None:
        out = backend.staged(y, sr, cfg.speed) if cfg.staged else backend.stretch(y, sr, cfg.speed)
    else:
        out = backend.stretch_map(y, sr, tm)
    t["stretch"] = time.perf_counter() - t0

    if cancel_check and cancel_check():
        from .chunking import CancelledError
        raise CancelledError("Processing cancelled by user")

    step("post chain")
    t0 = time.perf_counter()
    out = post.run(out, sr, cfg.post)
    t["post"] = time.perf_counter() - t0

    target = len(y) / cfg.speed
    notes["duration_error_pct"] = round((len(out) - target) / target * 100, 3)

    if verbose:
        for k, v in t.items():
            print(f"  {k:<10} {v:6.2f}s")
        for k, v in notes.items():
            print(f"  {k:<22} {v}")

    return Result(audio=out, sr=sr, timings=t, notes=notes)
