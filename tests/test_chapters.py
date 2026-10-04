"""Tests for semantic chapter detection, Table of Contents generator, and endpoints."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app import chapters
from app.api import app, OUTPUT_DIR

client = TestClient(app)


def test_extract_chapter_title_from_text():
    title = chapters._extract_chapter_title_from_text("Well today we are going to explore quantum mechanics and relativity")
    assert "Quantum Mechanics" in title or "Explore Quantum" in title
    assert len(title.split()) <= 4

    fallback = chapters._extract_chapter_title_from_text("yeah okay well", default="DefaultTitle")
    assert fallback == "DefaultTitle"


def test_algorithmic_chapters_short_clip():
    segs = [
        {"start": 0.0, "end": 15.0, "text": "Introduction to neural network architectures."},
        {"start": 16.0, "end": 35.0, "text": "Deep residual learning for image recognition."},
    ]
    chs = chapters.generate_algorithmic_chapters(segs, duration_s=40.0)
    assert len(chs) == 1
    assert chs[0]["start"] == 0.0
    assert chs[0]["end"] == 40.0


def test_algorithmic_chapters_multi_minute():
    # Construct a 300-second session with a few pause gaps
    segs = []
    t = 0.0
    for i in range(30):
        # 8-second speech
        segs.append({
            "start": round(t, 2),
            "end": round(t + 8.0, 2),
            "text": f"Segment index {i} discussing speech compression mechanisms and algorithmic performance.",
        })
        # At indices 9 and 19, insert a 3.0 second pause gap
        gap = 3.0 if i in (9, 19) else 0.4
        t += 8.0 + gap

    total_dur = t
    chs = chapters.generate_algorithmic_chapters(segs, duration_s=total_dur, min_chapter_duration_s=45.0)
    assert len(chs) >= 2
    assert chs[0]["start"] == 0.0
    assert chs[-1]["end"] == round(total_dur, 2)

    # Monotonicity check
    for idx in range(len(chs) - 1):
        assert chs[idx]["end"] <= chs[idx + 1]["start"] + 0.1
        assert chs[idx]["title"]


def test_detect_and_save_chapters_caching(tmp_path):
    segs = [
        {"start": 0.0, "end": 10.0, "text": "First chapter content."},
        {"start": 12.0, "end": 60.0, "text": "Second chapter content."},
    ]
    chs = chapters.detect_and_save_chapters(
        output_name="test_speech_5x.wav",
        segments=segs,
        duration_s=60.0,
        output_dir=tmp_path,
        use_llm=False,
    )
    assert len(chs) >= 1
    cached_file = tmp_path / "test_speech_5x.chapters.json"
    assert cached_file.is_file()

    loaded = chapters.load_cached_chapters("test_speech_5x.wav", output_dir=tmp_path)
    assert loaded is not None
    assert len(loaded) == len(chs)
    assert loaded[0]["title"] == chs[0]["title"]


def test_query_llm_chapters_fallback_on_error(monkeypatch):
    # Simulate LLM router down
    def fake_open(*args, **kwargs):
        raise ConnectionRefusedError("LLM router offline")

    monkeypatch.setattr(chapters.urllib.request, "urlopen", fake_open)
    segs = [{"start": 0.0, "end": 80.0, "text": "Audio segment"}]
    res = chapters.query_llm_chapters(segs, duration_s=80.0)
    assert res is None


def test_chapters_parse_json_response():
    from app.chapters import _parse_json_response

    res = _parse_json_response('```json\n{"chapters": [{"title": "Intro", "start": 0.0, "end": 30.0}]}\n```')
    assert res is not None
    assert "chapters" in res
    assert res["chapters"][0]["title"] == "Intro"


def test_query_llm_chapters_handles_markdown_fence(monkeypatch):
    from unittest.mock import MagicMock

    fenced_payload = {
        "choices": [{
            "message": {
                "content": "```json\n{\"chapters\": [{\"title\": \"Intro Section\", \"start\": 0.0, \"end\": 40.0}, {\"title\": \"Outro Section\", \"start\": 40.0, \"end\": 80.0}]}\n```"
            }
        }]
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(fenced_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    monkeypatch.setattr(chapters.urllib.request, "urlopen", lambda req, timeout: mock_resp)

    segs = [
        {"start": 0.0, "end": 40.0, "text": "Introduction part."},
        {"start": 40.0, "end": 80.0, "text": "Conclusion part."},
    ]
    res = chapters.query_llm_chapters(segs, duration_s=80.0, timeout_s=5.0)
    assert res is not None
    assert len(res) == 2
    assert res[0]["title"] == "Intro Section"
    assert res[1]["title"] == "Outro Section"


def test_chapters_api_endpoints(tmp_path, monkeypatch):
    stem = "api_chapter_test_output"
    audio_file = OUTPUT_DIR / f"{stem}.wav"
    audio_file.write_bytes(b"RIFF dummy wav data")

    transcript_file = OUTPUT_DIR / f"{stem}.synced.json"
    sample_data = {
        "output": f"{stem}.wav",
        "output_duration_s": 120.0,
        "segments": [
            {"start": 0.0, "end": 50.0, "text": "Introduction to astrophysics."},
            {"start": 52.0, "end": 120.0, "text": "Stellar evolution and supernova dynamics."},
        ]
    }
    transcript_file.write_text(json.dumps(sample_data), encoding="utf-8")

    try:
        # 1. GET chapters -> should generate and save
        resp = client.get(f"/api/v1/chapters/{stem}.wav?use_llm=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("generated", "cached")
        assert len(data["chapters"]) >= 1
        assert data["stem"] == stem

        # 2. GET again -> should hit cache
        resp_cached = client.get(f"/api/v1/chapters/{stem}.wav")
        assert resp_cached.status_code == 200
        assert resp_cached.json()["status"] == "cached"

        # 3. POST force regenerate
        resp_post = client.post(f"/api/v1/chapters/{stem}.wav", json={"use_llm": False, "force_regenerate": True})
        assert resp_post.status_code == 200
        assert resp_post.json()["status"] == "generated"
    finally:
        audio_file.unlink(missing_ok=True)
        transcript_file.unlink(missing_ok=True)
        (OUTPUT_DIR / f"{stem}.chapters.json").unlink(missing_ok=True)
