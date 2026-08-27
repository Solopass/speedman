import numpy as np
import pytest

from speedman.config import build_config
from speedman.pipeline import process
from speedman.stretch import get_backend

SR = 24000
rb = pytest.mark.skipif(not get_backend("rubberband").available(), reason="rubberband unavailable")


@rb
@pytest.mark.parametrize("speed", [3.0, 5.0, 6.0, 8.0])
def test_golden_smoke(speechlike, speed):
    res = process(speechlike, SR, build_config(speed=speed, preset="fast"))
    assert np.isfinite(res.audio).all()
    assert np.abs(res.audio).max() <= 0.9701, "clipping"
    assert abs(res.notes["duration_error_pct"]) < 2.0


@rb
def test_uniform_control_runs(speechlike):
    res = process(speechlike, SR, build_config(speed=5.0, uniform=True))
    assert abs(res.notes["duration_error_pct"]) < 2.0
    assert res.notes["mode"] == "uniform"


def test_non_timemap_backend_refuses_nonuniform(speechlike):
    cfg = build_config(speed=5.0, backend="phasevocoder")
    with pytest.raises(ValueError, match="cannot apply a time map"):
        process(speechlike, SR, cfg)


@rb
def test_presets_all_work(speechlike):
    for preset in ("natural", "fast", "aggressive", "max"):
        res = process(speechlike, SR, build_config(speed=6.0, preset=preset))
        assert np.isfinite(res.audio).all()
        assert abs(res.notes["duration_error_pct"]) < 2.0, preset
