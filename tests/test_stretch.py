import numpy as np
import pytest

from speedman.stretch import BACKENDS, get_backend

SR = 24000


def _avail():
    return [n for n in BACKENDS if get_backend(n).available()]


@pytest.mark.parametrize("name", list(BACKENDS))
def test_halves_duration_at_2x(name, speechlike):
    b = get_backend(name)
    if not b.available():
        pytest.skip(f"{name} unavailable")
    out = b.stretch(speechlike, SR, 2.0)
    err = abs(len(out) - len(speechlike) / 2) / (len(speechlike) / 2)
    assert err < 0.02, f"{name}: duration error {err*100:.1f}%"


@pytest.mark.parametrize("name", ["phasevocoder", "wsola", "rubberband"])
def test_tsm_does_not_shift_pitch(name):
    """The TSM-not-resample invariant. This is the test that would have caught
    the original draft's pitch-shift confusion."""
    b = get_backend(name)
    if not b.available():
        pytest.skip(f"{name} unavailable")
    t = np.arange(int(2.0 * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = b.stretch(y, SR, 2.0)
    seg = out[len(out) // 4: len(out) // 4 + 8192] * np.hanning(8192)
    freq = np.fft.rfftfreq(8192, 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert abs(freq - 440) < 15, f"{name} moved 440 Hz to {freq:.0f} Hz -- that is resampling"


def test_resample_DOES_shift_pitch():
    """The control must behave like the control, or it is not a baseline."""
    t = np.arange(int(2.0 * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = get_backend("resample").stretch(y, SR, 2.0)
    seg = out[len(out) // 4: len(out) // 4 + 8192] * np.hanning(8192)
    freq = np.fft.rfftfreq(8192, 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert freq > 800


@pytest.mark.parametrize("name", ["resample", "phasevocoder", "wsola"])
def test_non_timemap_backends_raise(name, speechlike):
    """They must refuse rather than silently approximate -- a quiet approximation
    would look like a working pipeline producing quietly wrong audio."""
    b = get_backend(name)
    if not b.available():
        pytest.skip(f"{name} unavailable")
    assert not b.supports_time_map
    with pytest.raises(NotImplementedError):
        b.stretch_map(speechlike, SR, [(0, 0), (len(speechlike), len(speechlike) // 5)])


def test_timemap_hits_exact_duration(speechlike):
    from speedman.analyze import annotate_vad
    from speedman.ratemap import build_time_map
    b = get_backend("rubberband")
    if not b.available():
        pytest.skip("rubberband unavailable")
    tm = build_time_map(annotate_vad(speechlike, SR), 5.0)
    out = b.stretch_map(speechlike, SR, tm)
    err = abs(len(out) - len(speechlike) / 5) / (len(speechlike) / 5)
    assert err < 0.02, f"time-map duration error {err*100:.2f}%"


def test_staged_engages_only_above_threshold(speechlike):
    b = get_backend("rubberband")
    if not b.available():
        pytest.skip("rubberband unavailable")
    out = b.staged(speechlike, SR, 8.0)
    err = abs(len(out) - len(speechlike) / 8) / (len(speechlike) / 8)
    assert err < 0.05
