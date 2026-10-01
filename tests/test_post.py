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


def test_highpass_attenuates_sub_bass():
    from speedman.post import highpass_filter

    t = np.linspace(0, 1.0, SR, endpoint=False)
    y_sub = np.sin(2 * np.pi * 30 * t).astype(np.float32)
    y_voice = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    filtered_sub = highpass_filter(y_sub, SR, cutoff=80.0)
    filtered_voice = highpass_filter(y_voice, SR, cutoff=80.0)

    # 30 Hz must be suppressed by at least 20 dB
    sub_loss_db = 20 * np.log10((np.std(filtered_sub) + 1e-9) / (np.std(y_sub) + 1e-9))
    assert sub_loss_db < -20.0

    # 300 Hz must pass virtually unattenuated (< 0.5 dB loss)
    voice_loss_db = 20 * np.log10((np.std(filtered_voice) + 1e-9) / (np.std(y_voice) + 1e-9))
    assert abs(voice_loss_db) < 0.5


def test_warmth_eq_lifts_vocal_body(speechlike):
    from speedman.post import warmth_eq

    out = warmth_eq(speechlike, SR, PostConfig(warmth_db=3.0, warmth_freq=280.0))
    f = np.fft.rfftfreq(len(speechlike), 1 / SR)
    band = (f > 200) & (f < 360)
    before = np.abs(np.fft.rfft(speechlike))[band].mean()
    after = np.abs(np.fft.rfft(out))[band].mean()
    assert after > before * 1.1


def test_deess_tames_sibilance_burst():
    from speedman.post import deess

    t = np.linspace(0, 1.0, SR, endpoint=False)
    # Mixture of 1 kHz vowel and sharp 7 kHz sibilance burst
    y = (np.sin(2 * np.pi * 1000 * t) + 0.8 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)

    cfg = PostConfig(deess_db=5.0, deess_lo=5500.0, deess_hi=8500.0)
    out = deess(y, SR, cfg)

    f = np.fft.rfftfreq(len(y), 1 / SR)
    sib_band = (f > 6000) & (f < 8000)
    vowel_band = (f > 800) & (f < 1200)

    # 7 kHz sibilance should be reduced
    sib_before = np.abs(np.fft.rfft(y))[sib_band].max()
    sib_after = np.abs(np.fft.rfft(out))[sib_band].max()
    assert sib_after < sib_before

    # 1 kHz vowel should remain virtually unchanged
    vowel_before = np.abs(np.fft.rfft(y))[vowel_band].max()
    vowel_after = np.abs(np.fft.rfft(out))[vowel_band].max()
    assert abs(20 * np.log10(vowel_after / vowel_before)) < 0.5


def test_audiophile_preset_runs_cleanly(speechlike):
    from speedman.config import build_config
    from speedman.pipeline import process

    cfg = build_config(speed=3.0, preset="audiophile")
    assert cfg.engine == "r3"
    assert cfg.post.deess_db == 4.0
    assert cfg.post.highpass_hz == 80.0
    assert cfg.post.warmth_db == 2.0

    res = process(speechlike, SR, cfg)
    assert np.isfinite(res.audio).all()
    assert np.abs(res.audio).max() <= 0.9701
    assert len(res.audio) < len(speechlike)

