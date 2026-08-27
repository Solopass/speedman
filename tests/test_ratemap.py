import numpy as np
import pytest

from speedman.analyze import Annotation, Span, SpanKind, annotate_vad
from speedman.config import RateMapConfig
from speedman.ratemap import build_time_map

SR = 24000


def _ann(spec):
    """spec: [(kind, seconds), ...]"""
    spans, pos = [], 0
    for kind, dur in spec:
        n = int(dur * SR)
        spans.append(Span(pos, pos + n, kind))
        pos += n
    return Annotation(spans=spans, sr=SR, n_samples=pos)


@pytest.mark.parametrize("speed", [2.0, 3.0, 5.0, 8.0, 10.0])
def test_average_honours_requested_speed(speechlike, speed):
    ann = annotate_vad(speechlike, SR)
    tm = build_time_map(ann, speed)
    err = abs(tm.n_out - ann.n_samples / speed) / (ann.n_samples / speed)
    assert err < 0.02, f"duration error {err*100:.2f}% at {speed}x"


def test_map_is_strictly_monotonic(speechlike):
    build_time_map(annotate_vad(speechlike, SR), 6.0).check_monotonic()


def test_boundary_pauses_are_floored(speechlike):
    cfg = RateMapConfig(floor_ms=25.0, boundary_ms=250.0)
    ann = _ann([(SpanKind.SPEECH, 1.0), (SpanKind.SILENCE, 0.5), (SpanKind.SPEECH, 1.0)])
    tm = build_time_map(ann, 10.0, cfg)
    a, b = int(1.0 * SR), int(1.5 * SR)
    out_ms = (tm.expected_output(np.array([b])) - tm.expected_output(np.array([a])))[0] / SR * 1000
    assert out_ms >= 24.0, f"boundary pause survived only {out_ms:.1f}ms, floor is 25ms"


def test_short_gaps_are_NOT_floored():
    """The inversion guard. Flooring every gap makes lever 1 net-harmful in the
    4-8x band, because floored pauses then eat more than their uniform share."""
    cfg = RateMapConfig(floor_ms=25.0, boundary_ms=250.0)
    # 40 short 120ms gaps -- under the boundary threshold
    spec = []
    for _ in range(40):
        spec += [(SpanKind.SPEECH, 0.30), (SpanKind.SILENCE, 0.12)]
    ann = _ann(spec)
    tm = build_time_map(ann, 10.0, cfg)
    a, b = int(0.30 * SR), int(0.42 * SR)
    out_ms = (tm.expected_output(np.array([b])) - tm.expected_output(np.array([a])))[0] / SR * 1000
    assert out_ms < 20.0, (
        f"short gap kept {out_ms:.1f}ms -- it was floored, which is the inversion bug")
    assert abs(tm.n_out - ann.n_samples / 10.0) / (ann.n_samples / 10.0) < 0.02


def test_infeasible_constraints_still_hit_duration():
    """Many boundaries at extreme speed: the floor cannot be honoured. Duration
    accuracy is a hard criterion, the floor is a preference -- so the floor gives."""
    spec = []
    for _ in range(60):
        spec += [(SpanKind.SPEECH, 0.20), (SpanKind.SILENCE, 0.30)]
    ann = _ann(spec)
    # 60 boundaries x 25 ms floor = 1.5 s, but the whole target at 25x is 1.2 s.
    tm = build_time_map(ann, 25.0, RateMapConfig())
    err = abs(tm.n_out - ann.n_samples / 25.0) / (ann.n_samples / 25.0)
    assert err < 0.02, f"duration missed by {err*100:.2f}% under infeasible constraints"
    assert "floor_relaxations" in tm.notes or "floor_abandoned" in tm.notes


def test_rates_stay_within_bounds(speechlike):
    cfg = RateMapConfig(max_rate=40.0, min_rate=0.25)
    tm = build_time_map(annotate_vad(speechlike, SR), 10.0, cfg)
    assert tm.rates.max() <= cfg.max_rate * 1.001
    assert tm.rates.min() >= cfg.min_rate * 0.999


def test_lever2_is_duration_neutral(speechlike):
    """Raising protection cannot make the file shorter -- normalisation pins the
    average. If this ever fails, the map is not being normalised."""
    ann = annotate_vad(speechlike, SR)
    weak = build_time_map(ann, 6.0, RateMapConfig(protect_mult=0.95, crush_mult=1.1))
    strong = build_time_map(ann, 6.0, RateMapConfig(protect_mult=0.5, crush_mult=3.0))
    assert abs(weak.n_out - strong.n_out) / weak.n_out < 0.01
