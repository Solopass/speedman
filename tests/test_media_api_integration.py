"""Tests for Media API client and library integration."""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.api import app
from app.media_api import list_media_library, check_media_api_online

client = TestClient(app)


def test_media_api_status():
    resp = client.get("/api/v1/media/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "online" in data
    assert data["url"] == "http://127.0.0.1:8080"


def test_media_library_listing():
    resp = client.get("/api/v1/media/library")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    for it in items:
        assert "filename" in it
        assert "path" in it
        assert "windows_path" in it
        assert "size_mb" in it


def test_ingest_url_validation():
    # Invalid or unreachable URL should raise 400
    resp = client.post("/api/v1/ingest/url", json={
        "url": "https://invalid-non-existent-domain-999.com/video",
        "speed": 5.0,
    })
    assert resp.status_code == 400
    assert "Failed to extract audio" in resp.json()["detail"]


def test_transcribe_file_not_found():
    resp = client.post("/api/v1/transcribe/non_existent_file_999.wav", json={"engine": "parakeet"})
    assert resp.status_code == 404


def test_ingest_url_rejects_non_http_urls():
    """yt-dlp takes the URL positionally, so a leading '-' would be parsed as an option."""
    from app.media_api import extract_audio_from_url

    for bad in ("--exec=touch /tmp/pwned", "-J", "file:///etc/passwd", "ftp://x/y", ""):
        with pytest.raises(ValueError):
            extract_audio_from_url(bad)


def test_ingest_url_route_rejects_argument_injection():
    resp = client.post("/api/v1/ingest/url", json={
        "url": "--exec=touch /tmp/speedman_pwned", "speed": 5.0,
    })
    assert resp.status_code == 400
    assert not Path("/tmp/speedman_pwned").exists()
