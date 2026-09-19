"""Tests for long-audio chunking and boundary pause planning."""
import numpy as np
import pytest

from speedman.chunking import plan_chunks, find_silence_cutpoint, process_chunked, CancelledError
from speedman.config import build_config


def test_plan_chunks_short_audio():
    # 5-minute audio should not be split
    sr = 24000
    n_samples = sr * 60 * 5
    chunks = plan_chunks(n_samples, sr, target_chunk_min=25.0)
    assert len(chunks) == 1
    assert chunks[0] == (0, n_samples)


def test_plan_chunks_multi_block():
    # 60-minute audio with 25-minute target should yield 3 chunks
    sr = 24000
    n_samples = sr * 60 * 60
    chunks = plan_chunks(n_samples, sr, target_chunk_min=25.0)
    assert len(chunks) >= 2
    # Verify continuity
    assert chunks[0][0] == 0
    assert chunks[-1][1] == n_samples
    for a, b in zip(chunks[:-1], chunks[1:]):
        assert a[1] == b[0]


def test_find_silence_cutpoint_snapping():
    sr = 24000
    # Create 4 seconds: 1s sound, 1s pause, 2s sound
    t = np.linspace(0, 4.0, 4 * sr, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)
    y[sr:2*sr] = 0.0  # Silence from sample 24000 to 48000

    nominal = int(1.8 * sr)  # Nominal cutpoint is near 1.8s
    cut = find_silence_cutpoint(y, sr, nominal_sample=nominal, window_start_sample=0)

    # Cut should snap inside the silence region
    assert sr <= cut <= 2 * sr


def test_process_chunked_synthetic():
    sr = 24000
    # 6 seconds synthetic audio, forced into 2-second chunks
    t = np.linspace(0, 6.0, 6 * sr, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)
    # Put silence gaps at 2s and 4s
    y[int(1.9 * sr):int(2.2 * sr)] = 0.0
    y[int(3.9 * sr):int(4.2 * sr)] = 0.0

    cfg = build_config(speed=4.0, preset="fast", backend="rubberband")

    progress_events = []
    def on_prog(data):
        progress_events.append(data)

    out, timings, notes = process_chunked(
        y, sr, cfg,
        target_chunk_min=0.033,  # ~2 seconds
        on_progress=on_prog,
    )

    assert len(out) > 0
    assert notes["chunked"] is True
    assert notes["total_chunks"] >= 2
    assert len(progress_events) >= 2
    # Check approximate 4x duration (6s / 4 = 1.5s)
    out_dur = len(out) / sr
    assert 1.2 <= out_dur <= 1.8


def test_process_chunked_cancellation():
    sr = 24000
    t = np.linspace(0, 4.0, 4 * sr, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)
    cfg = build_config(speed=4.0, preset="fast", backend="rubberband")

    # Cancel check that returns True immediately on second chunk
    calls = 0
    def cancel_now():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(CancelledError):
        process_chunked(
            y, sr, cfg,
            target_chunk_min=0.016,  # 1-second chunks
            cancel_check=cancel_now,
        )
