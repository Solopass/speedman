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


def highpass_filter(y: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """4th-order Butterworth high-pass filter using SOS. Eliminates rumble & plosives."""
    if cutoff <= 0 or cutoff >= sr / 2:
        return y
    sos = butter(4, cutoff, btype="hp", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def warmth_eq(y: np.ndarray, sr: int, cfg: PostConfig) -> np.ndarray:
    """Additive low-mid vocal warmth peaking EQ at ~280 Hz to counterbalance presence."""
    if cfg.warmth_db == 0:
        return y
    gain = float(np.clip(cfg.warmth_db, -6.0, 6.0))
    f0 = float(np.clip(cfg.warmth_freq, 100.0, min(800.0, sr / 2 - 100.0)))
    q = float(max(0.2, cfg.warmth_q))
    b, a = peaking_biquad(sr, f0, q, gain)
    return lfilter(b, a, y).astype(np.float32)


def deess(y: np.ndarray, sr: int, cfg: PostConfig) -> np.ndarray:
    """Split-band dynamic sibilance tamer targeting 5.5-8.5 kHz fricatives.

    Fast attack (1 ms) prevents harsh piercing 's' bursts at speed, while
    leaving vowels and low-frequency speech completely untouched.
    """
    if cfg.deess_db <= 0:
        return y
    lo = float(max(1000.0, min(cfg.deess_lo, sr / 2 - 600.0)))
    hi = float(max(lo + 400.0, min(cfg.deess_hi, sr / 2 - 100.0)))
    if hi <= lo or lo >= sr / 2:
        return y

    sos_bp = butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos")
    sib_band = sosfilt(sos_bp, y)

    env_broad = _smooth_envelope(y, sr, attack_ms=2.0, release_ms=40.0)
    env_sib = _smooth_envelope(sib_band, sr, attack_ms=1.0, release_ms=25.0)

    ratio = env_sib / (env_broad + 1e-5)
    excess = np.maximum(0.0, ratio - 0.25)
    max_red = 10.0 ** (-float(np.clip(cfg.deess_db, 0.5, 12.0)) / 20.0)
    duck = 1.0 - (1.0 - max_red) * np.clip(excess * 3.0, 0.0, 1.0)

    sos_lp = butter(1, 60.0, btype="lp", fs=sr, output="sos")
    smooth_duck = np.clip(sosfilt(sos_lp, duck), max_red, 1.0)

    out = y - (1.0 - smooth_duck) * sib_band
    return out.astype(np.float32)


def downward_expand(y: np.ndarray, sr: int, cfg: PostConfig) -> np.ndarray:
    """Gentle downward expansion on pauses below -42 dB to quiet room hiss."""
    if cfg.expander_db <= 0:
        return y
    env = _smooth_envelope(y, sr, attack_ms=10.0, release_ms=80.0)
    env_db = 20 * np.log10(env + 1e-9)
    thresh_db = -42.0
    under = np.maximum(0.0, thresh_db - env_db)
    max_att = float(np.clip(cfg.expander_db, 0.0, 12.0))
    gain_db = -np.minimum(max_att, under * 0.4)
    return (y * 10 ** (gain_db / 20.0)).astype(np.float32)



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
    if cfg.highpass_hz > 0:
        y = highpass_filter(y, sr, cfg.highpass_hz)
    y = presence_eq(y, sr, cfg)
    if cfg.warmth_db != 0:
        y = warmth_eq(y, sr, cfg)
    if cfg.deess_db > 0:
        y = deess(y, sr, cfg)
    y = transient_enhance(y, sr, onset_env, cfg)
    y = compress_dynamics(y, sr, cfg)
    if cfg.expander_db > 0:
        y = downward_expand(y, sr, cfg)
    y = normalize_loudness(y, sr, cfg.target_lufs)
    y = limit(y, sr, cfg.limiter_ceiling)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_waveform_peaks(y: np.ndarray, num_bins: int = 120) -> list[float]:
    """Downsample audio signal amplitude to num_bins normalized peaks [0.0 - 1.0].

    Used to render real signal envelopes in the waveform visualizer without
    shipping megabytes of raw decoded PCM over HTTP.
    """
    num_bins = max(1, int(num_bins))
    y = np.asarray(y, dtype=np.float32)
    y = np.atleast_1d(np.squeeze(y))
    if y.ndim > 1:
        y = np.mean(y, axis=-1 if y.shape[-1] < y.shape[0] else 0)
    if y.ndim > 1:
        y = y.flatten()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return [0.0] * num_bins
    chunk_size = len(y) // num_bins
    if chunk_size == 0:
        peaks = np.interp(
            np.linspace(0, len(y) - 1, num_bins),
            np.arange(len(y)),
            np.abs(y),
        )
    else:
        truncated = y[:chunk_size * num_bins].reshape(num_bins, chunk_size)
        peaks = np.max(np.abs(truncated), axis=1)
    max_val = float(np.max(peaks)) if np.max(peaks) > 0 else 1.0
    return [round(float(np.clip(p / max_val, 0.0, 1.0)), 3) for p in peaks]


