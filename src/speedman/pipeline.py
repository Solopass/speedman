"""Orchestration: audio in, sped-up audio out."""
from __future__ import annotations

import time
from dataclasses import dataclass

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


MAX_MINUTES_UNCHUNKED = 90.0
"""Chunking is not built yet (CLAUDE.md open question). Rather than dying in an
allocator halfway through someone's audiobook, refuse up front and say why."""


def process(y: np.ndarray, sr: int, cfg: Config, verbose: bool = False,
            on_progress=None, annotation=None) -> Result:
    minutes = len(y) / sr / 60
    if minutes > MAX_MINUTES_UNCHUNKED:
        raise ValueError(
            f"this file is {minutes:.0f} minutes long, and speedman does not yet "
            f"process files over {MAX_MINUTES_UNCHUNKED:.0f} minutes in one pass "
            "(chunked processing is not built yet -- see CLAUDE.md).\n"
            "Split it first, e.g.:\n"
            "  ffmpeg -i input.mp3 -f segment -segment_time 3600 -c copy part%03d.mp3"
        )
    t = {}
    notes: dict = {}
    backend = get_backend(cfg.backend)

    def step(msg):
        if on_progress:
            on_progress(msg)

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
        # The annotation depends only on the audio, never on the requested speed
        # or preset, so callers rendering several variants of one clip can build
        # it once and pass it in.
        ann = annotation or analyze.annotate_vad(
            y, sr,
            protect_window_ms=cfg.ratemap.protect_window_ms,
            protect_lead_ms=cfg.ratemap.protect_lead_ms,
        )
        tm = ratemap.build_time_map(ann, cfg.speed, cfg.ratemap)
        notes.update(tm.notes)
        notes["mode"] = "non-uniform"
        notes["n_anchors"] = len(tm.anchors)
        # s predicts lever 1's payoff: speech ends up near N*(1-0.6s)
        s = tm.notes.get("silence_fraction", 0.0)
        notes["effective_speech_rate"] = round(cfg.speed * (1 - 0.6 * s), 2)
    t["analyze"] = time.perf_counter() - t0

    step("stretching")
    t0 = time.perf_counter()
    if tm is None:
        out = backend.staged(y, sr, cfg.speed) if cfg.staged else backend.stretch(y, sr, cfg.speed)
    else:
        out = backend.stretch_map(y, sr, tm)
    t["stretch"] = time.perf_counter() - t0

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
