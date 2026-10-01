import pytest
from fastapi.testclient import TestClient

from app.transcript import parse_subtitle_content, TranscriptError
from app.api import app, OUTPUT_DIR
from app import timemap_store
from speedman import ratemap


VTT_SAMPLE = """WEBVTT

1
00:00:01.000 --> 00:00:03.500
Hello and welcome to Speedman.

2
00:00:04.200 --> 00:00:08.000
This is high-speed speech compression.
"""

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Hello and <i>welcome</i> to Speedman.

2
00:00:04,200 --> 00:00:08,000
This is high-speed speech compression.
"""

JSON_SAMPLE = """{
  "segments": [
    {"start": 1.0, "end": 3.5, "text": "Hello and welcome to Speedman."},
    {"start": 4.2, "end": 8.0, "text": "This is high-speed speech compression."}
  ]
}"""


def test_parse_subtitle_vtt():
    segments = parse_subtitle_content(VTT_SAMPLE, "sub.vtt")
    assert len(segments) == 2
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 3.5
    assert segments[0]["text"] == "Hello and welcome to Speedman."
    assert segments[1]["start"] == 4.2
    assert segments[1]["end"] == 8.0


def test_parse_subtitle_srt():
    segments = parse_subtitle_content(SRT_SAMPLE, "sub.srt")
    assert len(segments) == 2
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 3.5
    assert segments[0]["text"] == "Hello and welcome to Speedman."


def test_parse_subtitle_json():
    segments = parse_subtitle_content(JSON_SAMPLE, "sub.json")
    assert len(segments) == 2
    assert segments[0]["text"] == "Hello and welcome to Speedman."


def test_parse_subtitle_invalid():
    with pytest.raises(TranscriptError):
        parse_subtitle_content("Not a valid subtitle file at all.", "test.txt")


def test_parse_subtitle_varied_milliseconds_and_hours():
    # 1-digit and 2-digit ms
    varied_srt = """1
00:01:05,4 --> 00:01:08,55
Testing varied ms digits.
"""
    segs = parse_subtitle_content(varied_srt, "sub.srt")
    assert len(segs) == 1
    assert segs[0]["start"] == 65.4
    assert segs[0]["end"] == 68.55

    # 4-digit ms and 3-digit hours
    long_vtt = """WEBVTT

100:00:05.1234 --> 100:00:09.5678
Super long recording.
"""
    segs2 = parse_subtitle_content(long_vtt, "sub.vtt")
    assert len(segs2) == 1
    assert segs2[0]["start"] == pytest.approx(360005.1234, abs=0.01)
    assert segs2[0]["end"] == pytest.approx(360009.5678, abs=0.01)


def test_parse_subtitle_whisper_words_json():
    whisper_json = """{
      "words": [
        {"word": "Speedman", "start": 0.5, "end": 1.2},
        {"word": "rocks", "start": 1.3, "end": 1.8}
      ]
    }"""
    segs = parse_subtitle_content(whisper_json, "whisper.json")
    assert len(segs) == 2
    assert segs[0]["text"] == "Speedman"
    assert segs[0]["start"] == 0.5
    assert segs[1]["text"] == "rocks"


def test_import_transcript_endpoint():
    client = TestClient(app)
    out_name = "test_import_output.mp3"
    sr = 24000
    n_samples = sr * 10
    # Save a timemap for this test output
    timemap_store.save(out_name, None, speed=2.0, sample_rate=sr)

    try:
        res = client.post(
            "/api/v1/transcript/import",
            json={
                "output_name": out_name,
                "content": VTT_SAMPLE,
                "filename": "subtitles.vtt",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "synced"
        assert data["segments"] == 2
        assert "vtt_url" in data
        assert "source_vtt_url" in data

        # Check warped JSON was written to disk
        stem = "test_import_output"
        synced_json = OUTPUT_DIR / f"{stem}.synced.json"
        assert synced_json.is_file()
        assert (OUTPUT_DIR / f"{stem}.source.vtt").is_file()
    finally:
        # Cleanup
        for ext in (".vtt", ".source.vtt", ".synced.json", ".timemap.npz"):
            p = OUTPUT_DIR / f"test_import_output{ext}"
            if p.exists():
                p.unlink()
