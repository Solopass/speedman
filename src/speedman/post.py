"""Output chain: clarity EQ, optional transient enhancement, DRC, loudness, limiter.

Two rules the original draft broke and this must not:
  * The presence EQ is ADDITIVE (a peaking biquad), never a bandpass that
    replaces the signal with one band.
  * NO pre-emphasis. It is an analysis trick; applied to output without
    de-emphasis it just makes everything harsh.

The limiter is not optional. EQ (+up to 4 dB) plus transient boost plus
normalisation to -16 LUFS will otherwise exceed full scale on ordinary speech.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, sosfilt, butter

from .config import PostConfig


def peaking_biquad(sr: int, f0: float, q: float, gain_db: float):
    """RBJ cookbook peaking EQ. Additive by construction: gain_db=0 -> identity."""
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    a_ = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return b / a_[0], a_ / a_[0]


def presence_eq(y: np.ndarray, sr: int, cfg: PostConfig) -> np.ndarray:
    """Bounded consonant-clarity lift in the 2-5 kHz region."""
    if cfg.presence_db == 0:
        return y
    gain = float(np.clip(cfg.presence_db, -4.0, 4.0))
    f0 = float(np.sqrt(cfg.presence_lo * cfg.presence_hi))
    q = f0 / max(1.0, cfg.presence_hi - cfg.presence_lo)
    b, a = peaking_biquad(sr, f0, q, gain)
    return lfilter(b, a, y).astype(np.float32)


def _smooth_envelope(y: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Attack/release follower on |y|, vectorised.

    Fast attack comes from a peak-hold (which also gives the limiter lookahead --
    the reduction is in place before the peak arrives); the slow release comes
    from a one-pole lowpass. The one-pole guarantees the result is ramped, and a
    ramped gain signal is what stops the chain clicking.

    Was a per-sample Python loop, which cost roughly a minute per hour of audio
    and made the tool look hung on a real podcast.
    """
    from scipy.ndimage import maximum_filter1d

    env = np.abs(y).astype(np.float32)
    hold = max(1, int(sr * attack_ms / 1000))
    if hold > 1:
        env = maximum_filter1d(env, size=2 * hold + 1, mode="nearest")
    a = float(np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0)))
    return lfilter([1 - a], [1.0, -a], env).astype(np.float32)


def transient_enhance(y: np.ndarray, sr: int, onset_env: np.ndarray | None,
                      cfg: PostConfig) -> np.ndarray:
    """Onset-gated lift with ramped gain. Off by default: it ships only if the
    eval table says it helps."""
    if cfg.transient_db == 0 or onset_env is None or len(onset_env) == 0:
        return y
    env = np.interp(np.arange(len(y)), np.linspace(0, len(y) - 1, len(onset_env)), onset_env)
    env = env / (np.max(env) + 1e-9)
    sos = butter(2, 30.0, "lp", fs=sr, output="sos")     # ramp the gate
    gate = sosfilt(sos, env)
    gain = 1.0 + (10 ** (cfg.transient_db / 20.0) - 1.0) * np.clip(gate, 0, 1)
    return (y * gain).astype(np.float32)


def compress_dynamics(y: np.ndarray, sr: int, cfg: PostConfig) -> np.ndarray:
    """Gentle downward compression.

    At speed, quiet unstressed syllables fall below the perceptual floor between
    loud stressed ones. Pulling the range in keeps them audible.
    """
    if cfg.drc_ratio <= 1.0:
        return y
    env = _smooth_envelope(y, sr, attack_ms=5.0, release_ms=120.0)
    env_db = 20 * np.log10(env + 1e-9)
    over = np.maximum(0.0, env_db - cfg.drc_threshold_db)
    gain_db = -over * (1.0 - 1.0 / cfg.drc_ratio)
    return (y * 10 ** (gain_db / 20.0)).astype(np.float32)


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln
    except ImportError:
        peak = float(np.max(np.abs(y))) or 1.0
        return (y / peak * 0.7).astype(np.float32)
    if len(y) < sr * 0.4:                     # meter needs ~400 ms
        return y
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y.astype(np.float64))
    if not np.isfinite(loudness):
        return y
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyln.normalize.loudness(y.astype(np.float64), loudness,
                                       target_lufs).astype(np.float32)


def limit(y: np.ndarray, sr: int, ceiling: float = 0.97) -> np.ndarray:
    """Ramped-gain limiter. Required -- see module docstring.

    Gain reduction is smoothed with a fast attack and slow release so it never
    steps between samples, and a running minimum gives it lookahead so the
    reduction is already in place when the peak arrives.
    """
    peak = np.abs(y)
    if peak.size == 0 or peak.max() <= ceiling:
        return y
    need = np.minimum(1.0, ceiling / (peak + 1e-9))
    look = max(1, int(sr * 0.002))            # 2 ms lookahead
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need.astype(np.float32), size=2 * look + 1, mode="nearest")
    gain = _smooth_envelope(1.0 - need, sr, attack_ms=1.0, release_ms=60.0)
    out = (y * (1.0 - gain)).astype(np.float32)
    over = np.abs(out).max()
    return (out * (ceiling / over)).astype(np.float32) if over > ceiling else out


def run(y: np.ndarray, sr: int, cfg: PostConfig,
        onset_env: np.ndarray | None = None) -> np.ndarray:
    y = presence_eq(y, sr, cfg)
    y = transient_enhance(y, sr, onset_env, cfg)
    y = compress_dynamics(y, sr, cfg)
    y = normalize_loudness(y, sr, cfg.target_lufs)
    y = limit(y, sr, cfg.limiter_ceiling)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
