"""Tests for in-place transcript cue editing, find-and-replace, and subtitle synchronization."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import transcript as transcript_sync
from app.api import app, OUTPUT_DIR

client = TestClient(app)


def _setup_mock_transcript(stem: str, base_dir: Path) -> dict:
    data = {
        "output": f"{stem}.wav",
        "speed": 5.0,
        "sample_rate": 24000,
        "source_duration_s": 25.0,
        "output_duration_s": 5.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "source_start": 0.0,
                "source_end": 12.5,
                "text": "Hello world and welcome to Speedman testing.",
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 0.4, "source_start": 0.0, "source_end": 2.0},
                    {"text": "world", "start": 0.4, "end": 0.8, "source_start": 2.0, "source_end": 4.0},
                ],
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 5.0,
                "source_start": 12.5,
                "source_end": 25.0,
                "text": "The speed is five times faster with high fidelity.",
                "words": [
                    {"text": "The", "start": 2.5, "end": 2.8, "source_start": 12.5, "source_end": 14.0},
                    {"text": "speed", "start": 2.8, "end": 3.2, "source_start": 14.0, "source_end": 16.0},
                ],
            }
        ]
    }
    transcript_sync._rewrite_transcript_files(stem, data, base_dir)
    return data


def test_edit_transcript_cue(tmp_path):
    stem = "test_edit_cue"
    _setup_mock_transcript(stem, tmp_path)

    res = transcript_sync.edit_transcript_cue(
        output_name=f"{stem}.wav",
        line_index=0,
        new_text="Hello universe and welcome to Speedman audio.",
        output_dir=tmp_path,
    )
    assert res["status"] == "success"
    assert res["line_index"] == 0
    assert res["segment"]["text"] == "Hello universe and welcome to Speedman audio."

    # Check that word timings were re-estimated
    words = res["segment"]["words"]
    assert len(words) == 7
    assert words[1]["text"] == "universe"

    # Verify disk files were synchronized
    synced_data = json.loads((tmp_path / f"{stem}.synced.json").read_text(encoding="utf-8"))
    assert synced_data["segments"][0]["text"] == "Hello universe and welcome to Speedman audio."

    vtt_text = (tmp_path / f"{stem}.vtt").read_text(encoding="utf-8")
    assert "Hello universe and welcome to Speedman audio." in vtt_text

    txt_text = (tmp_path / f"{stem}.txt").read_text(encoding="utf-8")
    assert "Hello universe and welcome to Speedman audio." in txt_text


def test_edit_transcript_cue_invalid_index(tmp_path):
    stem = "test_invalid_cue"
    _setup_mock_transcript(stem, tmp_path)

    with pytest.raises(transcript_sync.TranscriptError, match="out of range"):
        transcript_sync.edit_transcript_cue(f"{stem}.wav", line_index=99, new_text="Fail", output_dir=tmp_path)


def test_batch_find_and_replace(tmp_path):
    stem = "test_find_replace"
    _setup_mock_transcript(stem, tmp_path)

    res = transcript_sync.batch_find_and_replace(
        output_name=f"{stem}.wav",
        find_text="speed",
        replace_text="velocity",
        match_case=False,
        output_dir=tmp_path,
    )
    assert res["status"] == "success"
    assert res["replacements_count"] >= 1
    assert 1 in res["modified_cues"]

    synced_data = json.loads((tmp_path / f"{stem}.synced.json").read_text(encoding="utf-8"))
    assert "velocity is five times" in synced_data["segments"][1]["text"].lower()

    srt_text = (tmp_path / f"{stem}.srt").read_text(encoding="utf-8")
    assert "velocity is five times" in srt_text.lower()


def test_transcript_edit_api_routes():
    stem = "api_transcript_edit_test"
    _setup_mock_transcript(stem, OUTPUT_DIR)

    try:
        # 1. Edit single cue
        resp = client.post(
            f"/api/v1/transcript/{stem}.wav/edit-cue",
            json={"line_index": 1, "new_text": "Modified cue via API endpoint."}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["segment"]["text"] == "Modified cue via API endpoint."

        # 2. Find and replace
        resp_fr = client.post(
            f"/api/v1/transcript/{stem}.wav/find-replace",
            json={"find": "API endpoint", "replace": "FastAPI service", "match_case": False}
        )
        assert resp_fr.status_code == 200
        fr_data = resp_fr.json()
        assert fr_data["replacements_count"] == 1

        vtt_content = (OUTPUT_DIR / f"{stem}.vtt").read_text(encoding="utf-8")
        assert "FastAPI service" in vtt_content
    finally:
        for ext in (".synced.json", ".vtt", ".source.vtt", ".srt", ".source.srt", ".txt"):
            (OUTPUT_DIR / f"{stem}{ext}").unlink(missing_ok=True)
