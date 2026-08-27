"""Annotation -> one global monotonic time map.

This is where both compression levers actually live, and where the two
non-obvious constraints from ARCHITECTURE.md are enforced:

  * Only BOUNDARY pauses get the floor. Flooring every gap makes lever 1
    net-harmful in the 4-8x band, because a floored pause then consumes MORE
    than its uniform share and pushes the speech faster than requested.
  * The map is normalised to hit the requested duration EXACTLY, which means
    lever 2 cannot make the file shorter -- it only reallocates. Raising
    protection widens `p` and forces everything else faster: wider protection
    is weaker protection.

Everything works at SPAN granularity, never per-sample: a 10-hour file is
~860M samples and per-sample float arrays would be gigabytes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analyze import Annotation, SpanKind
from .config import RateMapConfig


class InfeasibleRateMap(RuntimeError):
    pass


@dataclass
class TimeMap:
    anchors: np.ndarray          # (M, 2) int64: input sample -> output sample
    n_in: int
    n_out: int
    rates: np.ndarray            # (S,) instantaneous rate per segment, diagnostics
    seg_bounds: np.ndarray       # (S+1,) input sample boundaries
    notes: dict

    def as_pairs(self) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in self.anchors]

    def expected_output(self, in_samples: np.ndarray) -> np.ndarray:
        """Where the map says a given input sample should land. Used by tests and
        by the registration fixture."""
        return np.interp(in_samples, self.anchors[:, 0], self.anchors[:, 1])

    def check_monotonic(self) -> None:
        d = np.diff(self.anchors[:, 1])
        if np.any(d <= 0):
            raise ValueError("time map is not strictly increasing in the output domain")
        if np.any(np.diff(self.anchors[:, 0]) <= 0):
            raise ValueError("time map is not strictly increasing in the input domain")


def _segments(ann: Annotation, cfg: RateMapConfig):
    """(length, relative multiplier, floorable) per span.

    `floorable` marks boundary pauses -- the only spans that get the floor.
    """
    floor_len = cfg.boundary_ms * ann.sr / 1000.0
    lengths, mults, floorable = [], [], []
    for s in ann.spans:
        if s.kind is SpanKind.SILENCE:
            is_boundary = s.length >= floor_len
            # Short inter-word gaps ride with the speech (the inversion guard).
            mults.append(cfg.silence_mult if is_boundary else 1.0)
            floorable.append(is_boundary)
        elif s.kind is SpanKind.ONSET:
            mults.append(cfg.protect_mult)
            floorable.append(False)
        elif s.kind is SpanKind.STEADY:
            mults.append(cfg.crush_mult)
            floorable.append(False)
        else:
            mults.append(1.0)
            floorable.append(False)
        lengths.append(s.length)
    return (np.asarray(lengths, dtype=np.float64),
            np.asarray(mults, dtype=np.float64),
            np.asarray(floorable, dtype=bool))


def _solve_rates(lengths, mults, floorable, target_out, cfg, floor_samples):
    """Constrained water-filling.

    Normalising is not a single division: floored pauses and rate-clamped
    segments have FIXED output lengths, so removing them from the pool changes
    the scale factor for everything else, which can push new segments into
    violation. Iterate to a fixed point.
    """
    n_seg = len(lengths)
    fixed_out = np.full(n_seg, np.nan)          # NaN => still free

    for _ in range(64):
        free = np.isnan(fixed_out)
        remaining = target_out - np.nansum(fixed_out)
        if remaining <= 0 or not free.any():
            raise InfeasibleRateMap(
                f"fixed-length segments already consume {np.nansum(fixed_out):.0f} "
                f"samples of a {target_out:.0f}-sample target")

        # k such that sum_free(length / (k*mult)) == remaining
        k = float(np.sum(lengths[free] / mults[free]) / remaining)
        rates = k * mults
        out_len = lengths / np.maximum(rates, 1e-9)

        newly = np.zeros(n_seg, dtype=bool)
        # 1. boundary pauses that would fall under the floor
        v = free & floorable & (out_len < floor_samples)
        fixed_out[v] = floor_samples
        newly |= v
        # 2. instantaneous rate ceiling / floor (backend tolerance)
        v = free & (rates > cfg.max_rate) & ~newly
        fixed_out[v] = lengths[v] / cfg.max_rate
        newly |= v
        v = free & (rates < cfg.min_rate) & ~newly
        fixed_out[v] = lengths[v] / cfg.min_rate
        newly |= v

        if not newly.any():
            out = np.where(np.isnan(fixed_out), out_len, fixed_out)
            return out, np.where(np.isnan(fixed_out), rates, lengths / np.maximum(out, 1e-9))

    raise InfeasibleRateMap("rate-map solver did not converge")


def build_time_map(ann: Annotation, speed: float, cfg: RateMapConfig | None = None,
                   uniform: bool = False) -> TimeMap:
    """Build the global time map for `speed`.

    On an infeasible constraint set the floor is relaxed progressively rather
    than the duration target being missed -- duration accuracy is a hard
    acceptance criterion, the floor is a quality preference.
    """
    cfg = cfg or RateMapConfig()
    n = ann.n_samples
    target_out = n / speed
    notes: dict = {"silence_fraction": round(ann.silence_fraction(), 4)}

    if uniform:
        lengths = np.array([float(n)])
        rates = np.array([speed])
        out_lens = lengths / rates
        bounds = np.array([0, n], dtype=np.int64)
    else:
        lengths, mults, floorable = _segments(ann, cfg)
        floor_samples = cfg.floor_ms * ann.sr / 1000.0

        attempt_floor = floor_samples
        for attempt in range(8):
            try:
                out_lens, rates = _solve_rates(lengths, mults, floorable, target_out,
                                               cfg, attempt_floor)
                break
            except InfeasibleRateMap:
                attempt_floor *= 0.75
                notes.setdefault("floor_relaxations", 0)
                notes["floor_relaxations"] += 1
        else:
            # Last resort: no flooring at all. Boundaries suffer; duration is exact.
            out_lens, rates = _solve_rates(lengths, mults, floorable, target_out, cfg, 0.0)
            notes["floor_abandoned"] = True
            attempt_floor = 0.0
        if attempt_floor != floor_samples:
            notes["effective_floor_ms"] = round(attempt_floor / ann.sr * 1000, 2)
        bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    out_pos = np.concatenate([[0.0], np.cumsum(out_lens)])
    anchors = _anchors(bounds, out_pos, cfg.anchor_ms, ann.sr, n, target_out)

    tm = TimeMap(anchors=anchors, n_in=n, n_out=int(anchors[-1, 1]),
                 rates=rates, seg_bounds=bounds, notes=notes)
    tm.check_monotonic()
    return tm


def _anchors(bounds, out_pos, anchor_ms, sr, n_in, target_out) -> np.ndarray:
    """Anchors at every segment boundary plus a regular grid inside long segments.

    10 ms is the measured optimum (docs/SPIKE_RESULTS.md): 20 ms roughly doubles
    placement error, and finer than 10 ms buys nothing because the residual is
    the backend's own windowing jitter, not our sampling of the map.
    """
    step = max(1, int(anchor_ms * sr / 1000))
    pts = set(int(b) for b in bounds)
    pts.update(range(0, n_in, step))
    pts.add(n_in)
    xs = np.array(sorted(pts), dtype=np.int64)

    ys = np.interp(xs, bounds, out_pos)
    ys = np.rint(ys).astype(np.int64)
    ys[-1] = int(round(target_out))          # pin the final anchor: exact duration
    ys[0] = 0

    # strict monotonicity, without disturbing the pinned endpoint
    for i in range(1, len(ys) - 1):
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1
    if ys[-1] <= ys[-2]:
        ys[-2] = ys[-1] - 1
    return np.stack([xs, ys], axis=1)
