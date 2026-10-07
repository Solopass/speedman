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


def test_validate_ingest_url_normalizes_and_sanitizes():
    from app.media_api import validate_ingest_url

    assert validate_ingest_url("https://youtu.be/abc") == "https://youtu.be/abc"
    assert validate_ingest_url("  https://youtu.be/abc  ") == "https://youtu.be/abc"
    assert validate_ingest_url("<https://www.youtube.com/watch?v=abc>") == "https://www.youtube.com/watch?v=abc"
    assert validate_ingest_url("'https://youtu.be/abc'") == "https://youtu.be/abc"
    assert validate_ingest_url("\"https://youtu.be/abc\"") == "https://youtu.be/abc"
    assert validate_ingest_url("youtu.be/abc") == "https://youtu.be/abc"
    assert validate_ingest_url("youtube.com/watch?v=abc") == "https://youtube.com/watch?v=abc"
    assert validate_ingest_url("www.youtube.com/watch?v=abc") == "https://www.youtube.com/watch?v=abc"


def test_ingest_url_validates_preset_and_format():
    good = "https://example.com/video"
    assert client.post("/api/v1/ingest/url",
                       json={"url": good, "preset": "nope"}).status_code == 400
    assert client.post("/api/v1/ingest/url",
                       json={"url": good, "format": "ogg"}).status_code == 400


def test_transcribe_file_not_found():
    """A name that is not a Speedman output gives no stem to search from, so the route
    asks for source_path rather than guessing. (A missing *named* source is a 404 --
    see test_transcribe_404s_when_the_named_source_is_missing.)"""
    resp = client.post("/api/v1/transcribe/non_existent_file_999.wav", json={"engine": "parakeet"})
    assert resp.status_code == 400
    assert "source_path" in resp.json()["detail"]


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


# --------------------------------------------------------------------------- yt-dlp discovery

def test_finds_the_yt_dlp_this_machine_actually_has():
    """The fallback searched only ~/.local/bin and /usr/local/bin, so on this workstation
    it could never fire -- and the error told the user to install something they already
    had, in media-api's venv."""
    from app.media_api import find_ytdlp, MEDIA_API_VENV_YTDLP

    found = find_ytdlp()
    if MEDIA_API_VENV_YTDLP.is_file():
        assert found == MEDIA_API_VENV_YTDLP, "should prefer the auto-updating copy"
    if found is not None:
        assert found.is_file()


def test_env_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "yt-dlp"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("SPEEDMAN_YTDLP", str(fake))
    from app.media_api import find_ytdlp

    assert find_ytdlp() == fake


def test_env_override_pointing_at_nothing_is_ignored(monkeypatch):
    """A stale env var should not disable the search entirely."""
    monkeypatch.setenv("SPEEDMAN_YTDLP", "/definitely/not/here/yt-dlp")
    from app.media_api import find_ytdlp, MEDIA_API_VENV_YTDLP

    found = find_ytdlp()
    if MEDIA_API_VENV_YTDLP.is_file():
        assert found == MEDIA_API_VENV_YTDLP


def test_error_names_the_env_var_when_nothing_is_found(monkeypatch):
    import app.media_api as m

    monkeypatch.setattr(m, "find_ytdlp", lambda: None)
    monkeypatch.setattr(m, "check_media_api_online", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="SPEEDMAN_YTDLP"):
        m.extract_audio_from_url("https://www.youtube.com/watch?v=x")


# --------------------------------------------------------------------------- Daily Auto-Update

def test_should_check_ytdlp_update(tmp_path, monkeypatch):
    import time
    import app.media_api as m

    fake_ts = tmp_path / ".ytdlp_last_update_check"
    monkeypatch.setattr(m, "_LAST_UPDATE_CHECK_FILE", fake_ts)

    # 1. No file exists -> should check
    assert m.should_check_ytdlp_update() is True

    # 2. File written just now -> should not check
    m.record_ytdlp_update_checked()
    assert m.should_check_ytdlp_update() is False

    # 3. File written 25 hours ago -> should check
    old_time = time.time() - (86400 + 3600)
    fake_ts.write_text(str(old_time), encoding="utf-8")
    assert m.should_check_ytdlp_update() is True


def test_ensure_ytdlp_updated_skips_when_recent(tmp_path, monkeypatch):
    import app.media_api as m

    fake_ts = tmp_path / ".ytdlp_last_update_check"
    monkeypatch.setattr(m, "_LAST_UPDATE_CHECK_FILE", fake_ts)
    m.record_ytdlp_update_checked()

    res = m.ensure_ytdlp_updated(force=False)
    assert res["status"] == "skipped"
    assert "within 24h" in res["message"]


def test_update_ytdlp_endpoint(monkeypatch):
    import app.media_api as m

    monkeypatch.setattr(m, "ensure_ytdlp_updated", lambda force=True: {"status": "success", "version": "test"})
    resp = client.post("/api/v1/tools/update-ytdlp?force=true")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_download_video_falls_back_locally_when_media_api_offline(tmp_path, monkeypatch):
    import app.media_api as m

    monkeypatch.setattr(m, "check_media_api_online", lambda *a, **k: False)

    fake_mp4 = tmp_path / "mock_video.mp4"
    fake_mp4.write_bytes(b"dummy mp4 content")

    def mock_local(url, on_progress=None):
        return fake_mp4, False

    monkeypatch.setattr(m, "_download_video_via_local_ytdlp", mock_local)

    path, pre_existed = m.download_video_from_url("https://www.youtube.com/watch?v=mock")
    assert path == fake_mp4
    assert pre_existed is False



class _Done:
    def __init__(self, out):
        self.stdout = out


@pytest.mark.parametrize("out,state,online", [
    ("active\nactive\n", "running", True),
    ("inactive\nactive\n", "ready", True),      # socket-activated: the first real request starts it
    ("inactive\nfailed\n", "down", False),
])
def test_media_api_state_asks_systemd_not_the_port(monkeypatch, out, state, online):
    """Probing Media API over HTTP starts it (socket activation), so on the workstation the
    online check asks systemd instead and never connects."""
    from app import media_api
    monkeypatch.setattr(media_api, "MEDIA_API_BASE", "http://127.0.0.1:8080")
    monkeypatch.setattr(media_api, "_state_cache", (0.0, "unknown"))
    calls = []
    monkeypatch.setattr(media_api, "subprocess", type("S", (), {
        "run": staticmethod(lambda cmd, **k: calls.append(cmd) or _Done(out)),
        "SubprocessError": Exception}))

    def no_http(*a, **k):
        raise AssertionError("must not connect to Media API")
    monkeypatch.setattr(media_api.httpx, "Client", no_http)
    assert media_api.media_api_state(force=True) == state
    assert media_api.check_media_api_online(force=True) is online
    assert calls[0] == ["systemctl", "is-active", "media-api.service", "media-api.socket"]


def test_media_api_state_is_unknown_for_a_custom_url(monkeypatch):
    from app import media_api
    monkeypatch.setattr(media_api, "MEDIA_API_BASE", "http://192.168.1.5:8080")
    assert media_api.media_api_state(force=True, run=lambda *a, **k: _Done("active\nactive\n")) == "unknown"
