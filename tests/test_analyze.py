import numpy as np
from speedman.analyze import SpanKind, annotate_vad, detect_speech


def test_vad_finds_the_silence(tone_silence_tone, sr):
    mask = detect_speech(tone_silence_tone, sr)
    n = int(0.5 * sr)
    assert mask[: n - 1000].mean() > 0.9, "tone should read as speech"
    assert mask[n + 3000: n + int(0.6 * sr) - 3000].mean() < 0.1, "gap should read as silence"


def test_annotation_tiles_the_signal(speechlike, sr):
    ann = annotate_vad(speechlike, sr)
    ann.validate()                      # raises on gaps/overlaps
    assert sum(s.length for s in ann.spans) == len(speechlike)


def test_silence_fraction_is_plausible(speechlike, sr):
    ann = annotate_vad(speechlike, sr)
    assert 0.1 < ann.silence_fraction() < 0.6


def test_protection_windows_are_syllable_width(speechlike, sr):
    """Windows narrower than ~80 ms land on the wrong audio given the backend's
    ~5 ms placement jitter (docs/SPIKE_RESULTS.md)."""
    ann = annotate_vad(speechlike, sr, protect_window_ms=120.0)
    onsets = [s for s in ann.spans if s.kind is SpanKind.ONSET]
    assert onsets, "expected some protected regions"
    widths = np.array([s.duration(sr) * 1000 for s in onsets])
    assert np.median(widths) >= 80.0, f"median protection window {np.median(widths):.0f}ms < 80ms"
