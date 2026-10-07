import numpy as np
import pytest
from scipy.signal import butter, sosfilt

SR = 24000

# Nothing listens on the discard port, so a test can never reach the real Media API.
UNREACHABLE_MEDIA_API = "http://127.0.0.1:9"


@pytest.fixture(scope="session", autouse=True)
def no_live_media_api():
    """Keep the suite off the workstation's Media API (127.0.0.1:8080).

    Media API is socket-activated, so a test request starts it, and tests that ingest a URL
    leave jobs in its live history (D:\\AI\\Cache\\media_jobs.json): by 2026-10-07, 45 of
    its last 100 jobs were this suite's invalid-non-existent-domain-999.com downloads.
    Every call goes through app.media_api.MEDIA_API_BASE, so pointing it at a dead port
    makes Media API look offline, which is the path these tests are written for anyway.
    """
    from app import media_api

    real = media_api.MEDIA_API_BASE
    media_api.MEDIA_API_BASE = UNREACHABLE_MEDIA_API
    media_api._online_cache = (0.0, False)
    try:
        yield
    finally:
        media_api.MEDIA_API_BASE = real
        media_api._online_cache = (0.0, False)


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


@pytest.fixture
def synthetic_wav(tmp_path):
    """Creates a 1.0-second synthetic sine wave with a pause in tmp_path."""
    import soundfile as sf
    from pathlib import Path
    wav_path = tmp_path / "test_tone.wav"
    sr = SR
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)
    y[int(0.4 * sr):int(0.7 * sr)] = 0.0  # 300ms pause
    sf.write(str(wav_path), y, sr)
    return wav_path
