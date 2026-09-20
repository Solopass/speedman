"""Persist the time map beside each compressed output, so transcripts can be warped later.

`ratemap` solves a map for every compression and the pipeline then threw it away. Keeping
it is what turns an accurate 1x transcript into one that stays synced while you listen at
5x: naive `t / N` drifts 0.08-0.23s on real speech, because pauses compress harder than
speech and the drift accumulates wherever they fall.

Stored as .npz -- two int64 arrays, input sample bounds and output sample positions. A
35-minute podcast is ~16k segments, so this is tens of kilobytes, and warping is then an
np.interp exactly as `TimeMap.expected_output` does it.

Uniform runs have no map to store, only a speed. `warp` falls back to `t / speed`, which
is exact by definition in that mode rather than an approximation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.paths import is_within

logger = logging.getLogger("speedman_timemap")

OUTPUT_DIR = Path("/mnt/d/Audio/Speed")
TIMEMAP_DIR = OUTPUT_DIR / ".timemaps"


class TimeMapNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredTimeMap:
    speed: float
    sample_rate: int
    in_bounds: Optional[np.ndarray] = None
    out_positions: Optional[np.ndarray] = None

    @property
    def is_uniform(self) -> bool:
        return self.in_bounds is None or self.out_positions is None

    def warp(self, seconds):
        """Map input-timeline seconds onto the compressed timeline.

        Accepts a scalar or an array; returns the same shape. Values outside the file are
        clamped to its ends by np.interp, which is what a transcript segment running past
        the final sample should do anyway.
        """
        values = np.atleast_1d(np.asarray(seconds, dtype=np.float64))
        if self.is_uniform:
            out = values / self.speed
        else:
            samples = values * self.sample_rate
            out = np.interp(samples, self.in_bounds, self.out_positions) / self.sample_rate
        out = np.maximum(out, 0.0)
        return float(out[0]) if np.isscalar(seconds) or np.ndim(seconds) == 0 else out


def _path_for(output_name: str) -> Path:
    return TIMEMAP_DIR / f"{Path(str(output_name)).stem}.npz"


def save(output_name: str, time_map, speed: float, sample_rate: int) -> Path:
    """Record the map for a compressed output. Never fatal: losing a map costs a synced
    transcript, not the audio, so callers should not have to guard the call."""
    TIMEMAP_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(output_name)

    payload = {"speed": np.float64(speed), "sample_rate": np.int64(sample_rate)}
    if time_map is not None:
        anchors = np.asarray(time_map.anchors)
        payload["in_bounds"] = anchors[:, 0].astype(np.int64)
        payload["out_positions"] = anchors[:, 1].astype(np.int64)

    np.savez_compressed(path, **payload)
    return path


def load(output_name: str) -> StoredTimeMap:
    path = _path_for(output_name)
    if not path.is_file() or not is_within(path, TIMEMAP_DIR):
        raise TimeMapNotFound(
            f"no time map recorded for '{Path(str(output_name)).name}'. It is written at "
            "compression time, so anything produced before this feature existed has none."
        )

    with np.load(path) as data:
        return StoredTimeMap(
            speed=float(data["speed"]),
            sample_rate=int(data["sample_rate"]),
            in_bounds=data["in_bounds"] if "in_bounds" in data else None,
            out_positions=data["out_positions"] if "out_positions" in data else None,
        )


def save_quietly(output_name: str, time_map, speed: float, sample_rate: int) -> None:
    """save(), but a failure is logged rather than raised -- see the note on save()."""
    try:
        save(output_name, time_map, speed, sample_rate)
    except Exception as e:
        logger.warning(f"[timemap] could not record map for {output_name}: {e}")
