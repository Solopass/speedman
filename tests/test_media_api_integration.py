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


def test_ingest_url_queues_a_job_rather_than_downloading_inline():
    """The download moved into the job. A syntactically valid URL is accepted immediately
    even if the host turns out to be unreachable -- that failure surfaces on the job, not
    as a request that held the connection open while yt-dlp timed out."""
    resp = client.post("/api/v1/ingest/url", json={
        "url": "https://invalid-non-existent-domain-999.com/video",
        "speed": 5.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert body["source_url"].startswith("https://")
    assert client.get(f"/api/v1/jobs/{body['job_id']}").status_code == 200


def test_ingest_url_rejects_a_bad_scheme_immediately():
    """A URL that can never work should fail fast rather than becoming a doomed job."""
    for bad in ("--exec=touch /tmp/pwned", "file:///etc/passwd", "ftp://x/y", ""):
        resp = client.post("/api/v1/ingest/url", json={"url": bad, "speed": 5.0})
        assert resp.status_code in (400, 422), f"{bad!r} was accepted"


def test_ingest_url_validates_preset_and_format():
    good = "https://example.com/video"
    assert client.post("/api/v1/ingest/url",
                       json={"url": good, "preset": "nope"}).status_code == 400
    assert client.post("/api/v1/ingest/url",
                       json={"url": good, "format": "ogg"}).status_code == 400


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
