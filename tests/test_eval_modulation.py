"""Tests for the modulation-spectrum metric.

Synthetic signals with a known modulation rate: if the analyser cannot recover a rate it
was handed, nothing it says about speech is worth reading.
"""
import numpy as np
import pytest

from eval.modulation import (
    analyse,
    SYLLABIC_LO_HZ,
    SYLLABIC_HI_HZ,
)

SR = 24000
DURATION_S = 20


def modulated_noise(mod_hz: float, seed: int = 0, duration_s: int = DURATION_S) -> np.ndarray:
    """Broadband noise amplitude-modulated at `mod_hz`, so every octave band carries it."""
    t = np.arange(SR * duration_s) / SR
    carrier = np.random.RandomState(seed).randn(len(t)) * 0.1
    env = 0.5 * (1.0 + np.sin(2 * np.pi * mod_hz * t))
    return (carrier * env).astype(np.float32)


@pytest.mark.parametrize("mod_hz", [2.0, 4.0, 8.0, 16.0, 32.0])
def test_recovers_known_modulation_rate(mod_hz):
    assert analyse(modulated_noise(mod_hz), SR).peak_hz == pytest.approx(mod_hz, abs=0.3)


@pytest.mark.parametrize("speed", [2.0, 3.0, 5.0])
def test_rate_change_scales_modulation_frequency(speed):
    """Compressing time by N multiplies every modulation frequency by N. This is the
    whole premise: it is why the metric can see what speed does to speech."""
    base_hz = 4.0
    y = modulated_noise(base_hz, seed=1)
    idx = np.arange(0, len(y), speed)
    resampled = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    assert analyse(resampled, SR).peak_hz == pytest.approx(base_hz * speed, rel=0.05)


def test_syllabic_fraction_high_inside_band_low_outside():
    inside = analyse(modulated_noise(4.0), SR).syllabic_fraction
    outside = analyse(modulated_noise(50.0), SR).syllabic_fraction
    assert inside > 0.5
    assert outside < 0.2
    assert inside > outside


def test_centroid_rises_monotonically_with_modulation_rate():
    """Centroid is the statistic that correlated with WER (rho +0.88); peak_hz did not,
    being an argmax and therefore noisy. Guard the one that is load-bearing."""
    centroids = [analyse(modulated_noise(hz, seed=2), SR).centroid_hz
                 for hz in (2.0, 4.0, 8.0, 16.0, 32.0)]
    assert all(a < b for a, b in zip(centroids, centroids[1:])), centroids


def test_band_constants_are_the_speech_band():
    assert SYLLABIC_LO_HZ == 2.0
    assert SYLLABIC_HI_HZ == 16.0


def test_too_short_input_returns_nan_rather_than_a_wrong_number():
    """Under a second cannot resolve 2 Hz; silently returning a number would be worse."""
    stats = analyse(np.zeros(SR // 2, dtype=np.float32), SR)
    assert stats.syllabic_fraction != stats.syllabic_fraction  # NaN


def test_silence_does_not_crash_or_fabricate():
    stats = analyse(np.zeros(SR * 5, dtype=np.float32), SR)
    assert stats.syllabic_fraction != stats.syllabic_fraction


def test_as_dict_exposes_all_three_statistics():
    d = analyse(modulated_noise(4.0), SR).as_dict()
    assert set(d) == {"syllabic_fraction", "peak_hz", "centroid_hz"}
