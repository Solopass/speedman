"""Tests for time-map persistence and transcript warping.

The warp is the whole value: naive `t / N` drifts because pauses compress harder than
speech, and that drift accumulates with file length -- 0.2s on a 90-second clip but 4.2s
across a 35-minute podcast, which is a very visible subtitle offset.
"""
import json

import numpy as np
import pytest

from app import timemap_store, transcript as tsync
from app.transcript import TranscriptError, format_timestamp, to_vtt, warp_segments


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(timemap_store, "TIMEMAP_DIR", tmp_path / ".timemaps")
    monkeypatch.setattr(timemap_store, "OUTPUT_DIR", tmp_path)
    return tmp_path


class FakeMap:
    """Stands in for ratemap.TimeMap -- only `anchors` is read when storing."""

    def __init__(self, anchors):
        self.anchors = np.asarray(anchors, dtype=np.int64)


# --------------------------------------------------------------------------- storage

def test_round_trips_a_map(store):
    sr = 24000
    # 10s of input -> 2s of output, but the first half compresses harder than the second.
    anchors = [[0, 0], [5 * sr, 1 * sr // 2], [10 * sr, 2 * sr]]
    timemap_store.save("talk_5x_fast.wav", FakeMap(anchors), speed=5.0, sample_rate=sr)

    loaded = timemap_store.load("talk_5x_fast.wav")
    assert loaded.speed == 5.0
    assert loaded.sample_rate == sr
    assert not loaded.is_uniform
    # Halfway through the input lands a quarter of the way through the output.
    assert loaded.warp(5.0) == pytest.approx(0.5, abs=0.01)
    assert loaded.warp(10.0) == pytest.approx(2.0, abs=0.01)


def test_uniform_runs_store_only_a_speed_and_fall_back_to_division(store):
    """With no map there is nothing to interpolate, and t/N is exact by definition."""
    timemap_store.save("talk_5x_fast.wav", None, speed=5.0, sample_rate=24000)
    loaded = timemap_store.load("talk_5x_fast.wav")
    assert loaded.is_uniform
    assert loaded.warp(50.0) == pytest.approx(10.0)


def test_missing_map_says_why(store):
    with pytest.raises(timemap_store.TimeMapNotFound, match="compression time"):
        timemap_store.load("never_made.wav")


def test_lookup_is_by_stem_so_the_extension_does_not_matter(store):
    timemap_store.save("talk_5x_fast.wav", None, speed=4.0, sample_rate=24000)
    assert timemap_store.load("talk_5x_fast.mp3").speed == 4.0


def test_warp_accepts_arrays_and_stays_non_negative(store):
    timemap_store.save("t.wav", None, speed=2.0, sample_rate=24000)
    loaded = timemap_store.load("t.wav")
    out = loaded.warp(np.array([0.0, 10.0, 20.0]))
    assert np.allclose(out, [0.0, 5.0, 10.0])
    assert float(loaded.warp(-5.0)) >= 0.0


def test_save_quietly_swallows_failure(store, monkeypatch):
    """A lost map costs a synced transcript, not the audio, so it must never fail a job."""
    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(timemap_store, "save", boom)
    timemap_store.save_quietly("x.wav", None, 5.0, 24000)  # must not raise


# --------------------------------------------------------------------------- VTT format

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "00:00:00.000"),
    (1.003, "00:00:01.003"),
    (61.5, "00:01:01.500"),
    (3661.25, "01:01:01.250"),
    (-3.0, "00:00:00.000"),
])
def test_timestamp_formatting(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_vtt_has_a_header_and_numbered_cues():
    vtt = to_vtt([
        {"start": 1.0, "end": 2.0, "text": "first"},
        {"start": 2.0, "end": 3.5, "text": "second"},
    ])
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.000 --> 00:00:02.000" in vtt
    assert "\n1\n" in vtt and "\n2\n" in vtt
    assert "second" in vtt


# --------------------------------------------------------------------------- warping

def _stored(sr=24000):
    # 10s in -> 2s out, front half crushed harder.
    timemap_store.save("o.wav", FakeMap([[0, 0], [5 * sr, sr // 2], [10 * sr, 2 * sr]]),
                       speed=5.0, sample_rate=sr)
    return timemap_store.load("o.wav")


def test_warping_follows_the_map_not_the_average_rate(store):
    """The front half runs at 10x and the back half at 3.3x, so a naive t/5 is wrong in
    both directions. This is exactly the drift the map exists to remove."""
    stored = _stored()
    warped = warp_segments([{"start": 5.0, "end": 6.0, "text": "middle"}], stored)
    assert warped[0]["start"] == pytest.approx(0.5, abs=0.02)
    assert warped[0]["start"] != pytest.approx(5.0 / 5.0, abs=0.02)  # naive would say 1.0


def test_warping_keeps_the_source_times_for_reference(store):
    warped = warp_segments([{"start": 5.0, "end": 6.0, "text": "x"}], _stored())
    assert warped[0]["source_start"] == 5.0
    assert warped[0]["source_end"] == 6.0


def test_zero_length_cues_are_given_duration(store):
    """A segment sitting inside a hard-compressed pause can collapse to a single instant,
    and players silently drop a cue with no duration."""
    warped = warp_segments([{"start": 3.0, "end": 3.0, "text": "blip"}], _stored())
    assert warped[0]["end"] > warped[0]["start"]


def test_empty_and_malformed_segments_are_skipped(store):
    warped = warp_segments([
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": "bad", "end": 2.0, "text": "nope"},
        {"start": 3.0, "end": 4.0, "text": "kept"},
    ], _stored())
    assert len(warped) == 1
    assert warped[0]["text"] == "kept"


def test_warping_nothing_usable_is_an_error(store):
    with pytest.raises(TranscriptError, match="no usable segments"):
        warp_segments([{"start": 1.0, "end": 2.0, "text": ""}], _stored())


# --------------------------------------------------------------------------- end to end

def test_sync_writes_both_files(store, tmp_path):
    sr = 24000
    timemap_store.save("talk_5x_fast.mp3",
                       FakeMap([[0, 0], [5 * sr, sr // 2], [10 * sr, 2 * sr]]),
                       speed=5.0, sample_rate=sr)

    transcript = tmp_path / "talk.transcript.json"
    transcript.write_text(json.dumps({"segments": [
        {"start": 0.5, "end": 1.5, "text": "hello there"},
        {"start": 6.0, "end": 8.0, "text": "later on"},
    ]}), encoding="utf-8")

    result = tsync.sync_to_output(transcript, "talk_5x_fast.mp3", tmp_path)
    assert result.segment_count == 2
    assert result.vtt_path.name == "talk_5x_fast.vtt"
    assert result.json_path.name == "talk_5x_fast.synced.json"
    assert "hello there" in result.vtt_path.read_text(encoding="utf-8")

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["warped_with"] == "time_map"
    assert data["speed"] == 5.0
    assert [s["text"] for s in data["segments"]] == ["hello there", "later on"]


def test_sync_reports_a_missing_transcript(store, tmp_path):
    timemap_store.save("o.wav", None, speed=5.0, sample_rate=24000)
    with pytest.raises(TranscriptError, match="not found"):
        tsync.sync_to_output(tmp_path / "absent.json", "o.wav", tmp_path)


def test_sync_reports_a_transcript_with_no_segments(store, tmp_path):
    timemap_store.save("o.wav", None, speed=5.0, sample_rate=24000)
    bad = tmp_path / "empty.transcript.json"
    bad.write_text(json.dumps({"segments": []}), encoding="utf-8")
    with pytest.raises(TranscriptError, match="no segments"):
        tsync.sync_to_output(bad, "o.wav", tmp_path)


def test_vtt_is_served_as_text_vtt_not_audio():
    """Served by the audio route, so the extension has to be recognised -- a .vtt sent as
    audio/wav is silently ignored by <track>."""
    from pathlib import Path as P
    from app.api import get_audio_media_type

    assert get_audio_media_type(P("talk_5x_fast.vtt")) == "text/vtt"
    assert get_audio_media_type(P("talk_5x_fast.synced.json")) == "application/json"
    assert get_audio_media_type(P("talk_5x_fast.mp3")) == "audio/mpeg"
