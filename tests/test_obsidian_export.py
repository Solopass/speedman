"""Tests for Obsidian note export and summarization suite."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import obsidian
from app.api import app, OUTPUT_DIR
from app.obsidian import (
    generate_extractive_summary,
    generate_note_summary,
    format_obsidian_markdown,
    save_obsidian_note,
    ObsidianExportError,
)

client = TestClient(app)


def test_extractive_summary_generation():
    sample_text = (
        "Speedman is a high-speed speech compression and listening system. "
        "It achieves up to six times acceleration by applying non-uniform time-scale modification. "
        "Pauses are compressed aggressively while dense consonant clusters remain intelligible. "
        "Standard uniform division leads to severe distortion and cognitive fatigue over long sessions. "
        "The system incorporates de-essing and vocal warmth filtering to ensure acoustic clarity. "
        "In conclusion, Speedman allows speed-reading audio at supersonic velocity with high retention."
    )
    summary, takeaways = generate_extractive_summary(sample_text, max_sentences=2, max_takeaways=3)
    assert len(summary) > 20
    assert len(takeaways) >= 1
    assert any("Speedman" in t or "compression" in t for t in takeaways)


def test_note_summary_fallback_when_llm_fails(monkeypatch):
    monkeypatch.setattr(obsidian, "query_local_llm_summary", lambda text, timeout_s=12.0: None)
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to Speedman audio engineering."},
        {"start": 2.5, "end": 5.0, "text": "This episode covers time compression and phase vocoding."},
        {"start": 5.5, "end": 8.0, "text": "We retain full acoustic fidelity at five times playback speed."},
    ]
    res = generate_note_summary(segs, use_llm=True)
    assert "speedman-extractive" in res["engine"]
    assert res["summary"]
    assert len(res["takeaways"]) >= 1


def test_note_summary_uses_llm_when_available(monkeypatch):
    def fake_llm(text, timeout_s=12.0):
        return ("LLM Executive Summary: Advanced speech compression.", ["Takeaway 1: Fast", "Takeaway 2: Clear"])

    monkeypatch.setattr(obsidian, "query_local_llm_summary", fake_llm)
    segs = [{"start": 0.0, "end": 3.0, "text": "Hello world testing LLM path."}]
    res = generate_note_summary(segs, use_llm=True)
    assert "sol-fast" in res["engine"]
    assert "LLM Executive Summary" in res["summary"]
    assert len(res["takeaways"]) == 2


def test_format_obsidian_markdown():
    summary_data = {
        "summary": "This is the executive summary.",
        "takeaways": ["First major takeaway point", "Second major takeaway point"],
        "engine": "sol-fast",
    }
    bookmarks = [
        {
            "startTime": 12.0,
            "endTime": 17.0,
            "text": "This is an important quote marked with M hotkey.",
            "note": "Critical architectural decision",
        }
    ]
    segments = [
        {"start": 0.0, "source_start": 0.0, "text": "Introduction line."},
        {"start": 5.0, "source_start": 5.0, "text": "Second line."},
    ]

    md = format_obsidian_markdown(
        title="Podcast Episode 42",
        source_name="episode_42.wav",
        duration_s=120.0,
        speed=5.0,
        summary_data=summary_data,
        bookmarks=bookmarks,
        segments=segments,
        custom_tags=["ai", "architecture"],
        include_transcript=True,
    )

    assert "---" in md
    assert "made_by: speedman" in md
    assert "source: \"episode_42.wav\"" in md
    assert "tags:" in md
    assert "- speedman" in md
    assert "- architecture" in md
    assert "# 🎧 Podcast Episode 42" in md
    assert "> [!abstract] Executive Summary" in md
    assert "This is the executive summary." in md
    assert "> [!important] Key Takeaways" in md
    assert "- First major takeaway point" in md
    assert "## ⭐ Highlighted Quotes (1)" in md
    assert "This is an important quote marked with M hotkey." in md
    assert "[00:12 - 00:17]" in md
    assert "Critical architectural decision" in md
    assert "## 📜 Transcript Outline" in md
    assert "<details>" in md
    assert "Introduction line." in md


def test_save_obsidian_note(tmp_path):
    notebook = tmp_path / "1Notebook"
    notebook.mkdir()

    res = save_obsidian_note(
        note_title="Lecture_01",
        content="# Lecture 1\nContent goes here.",
        folder="Summaries",
        notebook_root=notebook,
    )
    assert res["status"] == "success"
    assert res["title"] == "Lecture_01"
    assert res["folder"] == "Summaries"
    assert (notebook / "Summaries" / "Lecture_01.md").is_file()
    assert "obsidian://open?vault=OBVLT&file=1Notebook%2FSummaries%2FLecture_01" in res["obsidian_uri"]


def test_save_obsidian_note_rejects_traversal(tmp_path):
    notebook = tmp_path / "1Notebook"
    notebook.mkdir()

    with pytest.raises(ObsidianExportError, match="outside the permitted"):
        save_obsidian_note(
            note_title="Hacked",
            content="evil",
            folder="../../etc",
            notebook_root=notebook,
        )


def test_api_export_obsidian_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", tmp_path / "1Notebook")
    (tmp_path / "1Notebook").mkdir()

    # Create dummy synced json
    stem = "test_lecture"
    test_json = OUTPUT_DIR / f"{stem}.synced.json"
    test_json.write_text(json.dumps({
        "segments": [{"start": 1.0, "end": 4.0, "source_start": 1.0, "text": "Speech synthesis."}],
        "source_duration_s": 60.0,
        "speed": 5.0,
        "source_path": f"{stem}.wav"
    }), encoding="utf-8")

    try:
        # Check folder list
        folders_resp = client.get("/api/v1/export/obsidian/folders")
        assert folders_resp.status_code == 200
        assert "Summaries" in folders_resp.json()["folders"]

        # Post export
        req_body = {
            "output_name": f"{stem}.wav",
            "folder": "Summaries",
            "note_title": "AI Speech Synthesis",
            "include_summary": True,
            "include_highlights": True,
            "include_transcript": True,
            "custom_tags": ["speech", "fast"],
            "bookmarks": [
                {"startTime": 1.0, "endTime": 4.0, "text": "Speech synthesis.", "note": "Important note"}
            ],
            "use_llm": False,
        }
        resp = client.post("/api/v1/export/obsidian", json=req_body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "success"
        assert "AI Speech Synthesis" in data["title"]
        assert Path(data["path"]).is_file()

        note_content = Path(data["path"]).read_text(encoding="utf-8")
        assert "Speech synthesis." in note_content
        assert "Executive Summary" in note_content
        assert "[00:01 - 00:04]" in note_content
    finally:
        if test_json.exists():
            test_json.unlink()
