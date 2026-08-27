import numpy as np

from speedman.config import PostConfig
from speedman.post import compress_dynamics, limit, presence_eq, run

SR = 24000


def test_eq_is_additive_at_zero_gain(speechlike):
    out = presence_eq(speechlike, SR, PostConfig(presence_db=0.0))
    assert np.allclose(out, speechlike)


def test_eq_lifts_the_presence_band(speechlike):
    out = presence_eq(speechlike, SR, PostConfig(presence_db=4.0))
    f = np.fft.rfftfreq(len(speechlike), 1 / SR)
    band = (f > 2000) & (f < 5000)
    before = np.abs(np.fft.rfft(speechlike))[band].mean()
    after = np.abs(np.fft.rfft(out))[band].mean()
    assert after > before * 1.1


def test_limiter_guarantees_ceiling():
    """Deliberately hot input -- the chain must not be able to clip."""
    rng = np.random.default_rng(0)
    hot = (rng.normal(0, 1, SR * 2) * 3.0).astype(np.float32)
    out = limit(hot, SR, ceiling=0.97)
    assert np.abs(out).max() <= 0.9701, f"peak {np.abs(out).max():.4f} exceeds ceiling"


def test_limiter_gain_is_ramped():
    """No sample-to-sample gain jumps -- that is what causes clicks."""
    y = np.concatenate([np.full(SR, 0.1), np.full(SR, 3.0)]).astype(np.float32)
    out = limit(y, SR, 0.97)
    ratio = np.abs(out[1:]) / (np.abs(y[1:]) + 1e-9)
    assert np.nanmax(np.abs(np.diff(ratio[SR - 200: SR + 2000]))) < 0.05


def test_drc_reduces_dynamic_range(speechlike):
    loud = speechlike * 2.0
    out = compress_dynamics(loud, SR, PostConfig(drc_threshold_db=-24, drc_ratio=4.0))
    assert np.std(np.abs(out)) < np.std(np.abs(loud))


def test_full_chain_is_finite_and_bounded(speechlike):
    out = run(speechlike * 2.5, SR, PostConfig(presence_db=4.0, transient_db=3.0))
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 0.9701
