import numpy as np
import pytest
from scipy.signal import butter, sosfilt

SR = 24000


@pytest.fixture(scope="session")
def sr():
    return SR


@pytest.fixture(scope="session")
def speechlike():
    """Voiced segments + consonant bursts + real pauses. Not speech, but carries
    the structures the pipeline keys on."""
    rng = np.random.default_rng(1)

    def voiced(dur, f0=120):
        n = int(dur * SR)
        t = np.arange(n) / SR
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 25))
        for f in (700, 1200, 2600):
            y += 0.6 * sosfilt(butter(2, [f * 0.85, f * 1.15], "bp", fs=SR, output="sos"), y)
        return y / np.max(np.abs(y)) * 0.6 * np.hanning(n) ** 0.3

    def burst(dur=0.02):
        n = int(dur * SR)
        return sosfilt(butter(2, [2500, 7000], "bp", fs=SR, output="sos"),
                       rng.normal(0, 1, n)) * np.hanning(n) * 0.8

    parts = []
    for _ in range(8):
        for _ in range(4):
            parts += [burst(), voiced(0.14 + 0.06 * rng.random())]
        parts.append(np.zeros(int((0.30 + 0.35 * rng.random()) * SR)))
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture(scope="session")
def tone_silence_tone():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return np.concatenate([tone, np.zeros(int(0.6 * SR), dtype=np.float32), tone])
