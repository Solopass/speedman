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
    def fake_llm(text, timeout_s=25.0):
        return ("LLM Executive Summary: Advanced speech compression.", ["Takeaway 1: Fast", "Takeaway 2: Clear"])

    monkeypatch.setattr(obsidian, "query_local_llm_summary", fake_llm)
    segs = [{"start": 0.0, "end": 3.0, "text": "Hello world testing LLM path."}]
    res = generate_note_summary(segs, use_llm=True)
    assert "sol-fast" in res["engine"]
    assert "LLM Executive Summary" in res["summary"]
    assert len(res["takeaways"]) == 2


def test_parse_json_response_markdown_fences():
    from app.obsidian import _parse_json_response

    # Clean JSON
    res1 = _parse_json_response('{"summary": "Test", "takeaways": ["Point 1"]}')
    assert res1 == {"summary": "Test", "takeaways": ["Point 1"]}

    # Markdown fence with ```json
    res2 = _parse_json_response('```json\n{"summary": "Fenced", "takeaways": ["Point 2"]}\n```')
    assert res2 == {"summary": "Fenced", "takeaways": ["Point 2"]}

    # Markdown fence without language tag
    res3 = _parse_json_response('```\n{"summary": "Plain fence", "takeaways": ["Point 3"]}\n```')
    assert res3 == {"summary": "Plain fence", "takeaways": ["Point 3"]}

    # Pre-amble thought text before JSON
    res4 = _parse_json_response('Here is your structured summary:\n{"summary": "With thought", "takeaways": ["Point 4"]}\nHope this helps!')
    assert res4 == {"summary": "With thought", "takeaways": ["Point 4"]}

    # Invalid string
    assert _parse_json_response('Not valid json at all') is None


def test_query_local_llm_summary_handles_markdown_fenced_response(monkeypatch):
    from unittest.mock import MagicMock
    import io

    fenced_payload = {
        "choices": [{
            "message": {
                "content": "```json\n{\"summary\": \"Robust summary parsed from fence.\", \"takeaways\": [\"A\", \"B\"]}\n```"
            }
        }]
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(fenced_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    monkeypatch.setattr(obsidian.urllib.request, "urlopen", lambda req, timeout: mock_resp)

    summary, takeaways = obsidian.query_local_llm_summary("Long transcript content for testing local AI fence handling...", timeout_s=5.0)
    assert summary == "Robust summary parsed from fence."
    assert takeaways == ["A", "B"]


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


def test_format_obsidian_with_audio_embed():
    md = format_obsidian_markdown(
        title="Test Talk",
        source_name="talk.wav",
        duration_s=120.0,
        speed=5.0,
        embed_audio=True,
        audio_filename="talk_5x_fast.wav",
    )
    assert "![[talk_5x_fast.wav]]" in md


def test_format_obsidian_with_deep_link_timestamps():
    bms = [{"startTime": 15.0, "endTime": 20.0, "text": "Crucial discovery."}]
    segs = [{"start": 15.0, "text": "Crucial discovery."}]
    md = format_obsidian_markdown(
        title="DeepLink Test",
        source_name="talk.wav",
        duration_s=60.0,
        speed=5.0,
        bookmarks=bms,
        segments=segs,
        deep_link_timestamps=True,
    )
    assert "obsidian://open?vault=OBVLT&file=DeepLink%20Test#t=15" in md


def test_append_quotes_to_daily_note(tmp_path, monkeypatch):
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", tmp_path / "1Notebook")
    (tmp_path / "1Notebook" / "Digests").mkdir(parents=True, exist_ok=True)

    quotes = [
        {"startTime": 10.0, "endTime": 15.0, "text": "Quote number one.", "note": "Key insight"},
        {"startTime": 20.0, "endTime": 25.0, "text": "Quote number two."},
    ]

    res = obsidian.append_quotes_to_daily_note(
        quotes=quotes,
        source_title="Podcast Ep 42",
        folder="Digests",
        date_str="2026-10-05",
    )
    assert res["status"] == "success"
    assert res["quotes_count"] == 2
    note_path = Path(res["path"])
    assert note_path.is_file()

    content = note_path.read_text(encoding="utf-8")
    assert "Podcast Ep 42" in content
    assert "Quote number one." in content
    assert "Quote number two." in content
    assert "[00:10 - 00:15]" in content

    # Append a second set of quotes to the same day
    res2 = obsidian.append_quotes_to_daily_note(
        quotes=[{"startTime": 30.0, "endTime": 35.0, "text": "Quote number three."}],
        source_title="Lecture 1",
        folder="Digests",
        date_str="2026-10-05",
    )
    assert res2["status"] == "success"
    updated_content = note_path.read_text(encoding="utf-8")
    assert "Lecture 1" in updated_content
    assert "Quote number three." in updated_content


def test_api_daily_digest_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", tmp_path / "1Notebook")
    (tmp_path / "1Notebook" / "Digests").mkdir(parents=True, exist_ok=True)

    payload = {
        "quotes": [{"startTime": 5.0, "endTime": 10.0, "text": "API quote test."}],
        "source_title": "Daily Briefing",
        "folder": "Digests",
        "date_str": "2026-10-05",
    }
    resp = client.post("/api/v1/export/obsidian/daily-digest", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_api_cache_stats_and_clean_endpoints(tmp_path, monkeypatch):
    from app import cache_cleaner
    # Use fake cache dirs in tmp_path
    downloads = tmp_path / "downloads"
    video_cache = tmp_path / "video-cache"
    comparisons = tmp_path / "comparisons"
    for d in (downloads, video_cache, comparisons):
        d.mkdir(parents=True)

    test_file = downloads / "stale_audio.wav"
    test_file.write_bytes(b"0" * 1024 * 100)  # 100 KB

    monkeypatch.setattr(cache_cleaner, "CACHE_ROOTS", [downloads, video_cache, comparisons])

    # 1. Check stats
    resp_stats = client.get("/api/v1/tools/cache-stats")
    assert resp_stats.status_code == 200
    data = resp_stats.json()
    assert data["total_files"] == 1
    assert data["total_bytes"] == 102400

    # 2. Clean cache with all_files=True
    resp_clean = client.post("/api/v1/tools/clean-cache", json={"all_files": True, "days": 0})
    assert resp_clean.status_code == 200
    clean_data = resp_clean.json()
    assert clean_data["deleted_count"] == 1
    assert not test_file.exists()


def test_save_obsidian_note_same_minute_collision(tmp_path, monkeypatch):
    notebook_dir = tmp_path / "1Notebook"
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", notebook_dir)

    # 1. First save
    res1 = obsidian.save_obsidian_note("Lecture Note", "Version 1 content", folder="Summaries", notebook_root=notebook_dir)
    assert Path(res1["path"]).is_file()
    assert Path(res1["path"]).read_text(encoding="utf-8") == "Version 1 content"

    # 2. Second save with different content in same minute -> must not overwrite Version 1
    res2 = obsidian.save_obsidian_note("Lecture Note", "Version 2 content", folder="Summaries", notebook_root=notebook_dir)
    assert Path(res2["path"]).is_file()
    assert res2["path"] != res1["path"]
    assert Path(res1["path"]).read_text(encoding="utf-8") == "Version 1 content"
    assert Path(res2["path"]).read_text(encoding="utf-8") == "Version 2 content"

    # 3. Third save with identical content -> should reuse without creating duplicate
    res3 = obsidian.save_obsidian_note("Lecture Note", "Version 1 content", folder="Summaries", notebook_root=notebook_dir)
    assert res3["path"] == res1["path"]


def test_append_quotes_to_daily_note_sanitizes_date_and_folder(tmp_path, monkeypatch):
    notebook_dir = tmp_path / "1Notebook"
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", notebook_dir)

    quotes = [{"startTime": 5.0, "endTime": 10.0, "text": "Sanitization quote."}]
    # Attempt directory traversal in date_str
    res = obsidian.append_quotes_to_daily_note(
        quotes=quotes,
        source_title="Traverse Test",
        folder="Digests",
        date_str="../../2026-10-05-bad",
        notebook_root=notebook_dir,
    )
    assert res["status"] == "success"
    # Slashes must be stripped from the date and resolved safely within Digests
    saved_path = Path(res["path"])
    assert saved_path.is_relative_to(notebook_dir / "Digests")
    assert ".." not in saved_path.name

