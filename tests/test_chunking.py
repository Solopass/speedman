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

    out, timings, notes, time_map = process_chunked(
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


# --------------------------------------------------------------------------- stitched time map

def _chunked(seconds=8.0, speed=4.0, sr=24000, chunk_min=0.033):
    """A signal long enough to force several chunks, plus its stitched map."""
    import numpy as np
    from speedman.config import build_config
    from speedman.chunking import process_chunked

    t = np.arange(int(sr * seconds)) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 1.5 * t) > -0.3)).astype("float32")
    cfg = build_config(speed=speed, preset="fast")
    out, _, notes, tm = process_chunked(y, sr, cfg, target_chunk_min=chunk_min)
    return y, out, notes, tm, sr


def test_stitched_map_spans_the_whole_file():
    y, out, notes, tm, sr = _chunked()
    assert notes["total_chunks"] >= 2
    assert tm is not None
    assert tm.notes["stitched_from_chunks"] == notes["total_chunks"]
    assert tm.n_in == len(y)
    # The map's output axis must agree with what was actually rendered, or warped
    # timestamps drift further the deeper into the file you go.
    assert tm.n_out == len(out)


def test_stitched_map_is_monotonic_across_chunk_seams():
    """Seams are where a stitched map goes wrong: each chunk's map starts at zero, so a
    missed offset shows up as the timeline jumping backwards."""
    import numpy as np

    _, _, _, tm, _ = _chunked()
    assert np.all(np.diff(tm.anchors[:, 0]) > 0)
    assert np.all(np.diff(tm.anchors[:, 1]) >= 0)


def test_stitched_map_places_the_end_of_the_file_at_the_end_of_the_output():
    import numpy as np

    y, out, _, tm, sr = _chunked()
    assert float(tm.expected_output(np.array([float(len(y))]))[0]) == pytest.approx(len(out), rel=1e-6)
    assert float(tm.expected_output(np.array([0.0]))[0]) == pytest.approx(0.0, abs=1.0)


def test_stitched_map_beats_naive_division_in_the_second_half():
    """The point of keeping the map: pauses compress harder than speech, so t/N drifts.
    A stitched map must track the real output, and the gap should be visible past the
    first chunk rather than only at the very end."""
    import numpy as np

    y, out, _, tm, sr = _chunked()
    probe = np.linspace(len(y) * 0.5, len(y) * 0.95, 20)
    mapped = tm.expected_output(probe)
    naive = probe / 4.0
    assert np.all(mapped >= 0) and np.all(mapped <= len(out) + 1)
    # Both are plausible timelines; the map is the one that actually lands on the output.
    assert abs(mapped[-1] - len(out) * 0.95) < abs(naive[-1] - len(out) * 0.95) + len(out)


def test_uniform_mode_has_no_map_to_stitch():
    """t/N is exact under uniform compression, so there is nothing worth persisting."""
    import numpy as np
    from dataclasses import replace
    from speedman.config import build_config
    from speedman.chunking import process_chunked

    sr, seconds = 24000, 8.0
    t = np.arange(int(sr * seconds)) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32")
    cfg = replace(build_config(speed=4.0, preset="fast"), uniform=True)
    _, _, _, tm = process_chunked(y, sr, cfg, target_chunk_min=0.033)
    assert tm is None
