"""Tests for Speedman FastAPI endpoints and session management."""
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pathlib import Path

from app.api import app, normalize_path, to_windows_path, is_path_allowed, OUTPUT_DIR, COMPARE_DIR

client = TestClient(app)


@pytest.fixture
def synthetic_wav(tmp_path) -> Path:
    """Creates a 1.0-second synthetic sine wave in tmp_path (which is under /tmp)."""
    wav_path = tmp_path / "test_tone.wav"
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    # 440 Hz tone with a pause
    y = np.sin(2 * np.pi * 440 * t)
    y[int(0.4 * sr):int(0.7 * sr)] = 0.0  # 300ms pause
    sf.write(str(wav_path), y, sr)
    return wav_path


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "speedman"
    assert data["version"] == "1.1.0"
    assert "windows_output_dir" in data


def test_root_dashboard_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Speedman" in resp.text
    assert "Ultra-Speed Speech Studio" in resp.text


def test_presets():
    resp = client.get("/api/v1/presets")
    assert resp.status_code == 200
    data = resp.json()
    for preset in ["natural", "fast", "aggressive", "max"]:
        assert preset in data
        assert "ratemap" in data[preset]
        assert "post" in data[preset]


def test_path_normalization():
    assert normalize_path("D:\\Audio\\Speed\\sample.wav") == Path("/mnt/d/Audio/Speed/sample.wav")
    assert normalize_path("d:/Workspace/speedman") == Path("/mnt/d/Workspace/speedman")
    assert normalize_path("/mnt/d/Audio/Speed") == Path("/mnt/d/Audio/Speed")
    assert to_windows_path("/mnt/d/Audio/Speed/test.wav") == "D:\\Audio\\Speed\\test.wav"


def test_path_security_allowlist(tmp_path):
    assert is_path_allowed(tmp_path) is True
    assert is_path_allowed(Path("/mnt/d/Audio")) is True
    assert is_path_allowed(Path("/etc/shadow")) is False
    assert is_path_allowed(Path("/root/.ssh/id_rsa")) is False


def test_compress_json_path(synthetic_wav):
    payload = {
        "input_path": str(synthetic_wav),
        "speed": 5.0,
        "preset": "fast",
        "uniform": False,
    }
    resp = client.post("/api/v1/compress/json", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["speed"] == 5.0
    assert data["preset"] == "fast"
    assert data["input_duration_s"] == 1.0
    assert 0.15 <= data["output_duration_s"] <= 0.25
    assert "audio_url" in data
    assert Path(data["output_path"]).is_file()


def test_compress_path_forbidden():
    payload = {
        "input_path": "/etc/passwd",
        "speed": 5.0,
        "preset": "fast",
    }
    resp = client.post("/api/v1/compress/json", json=payload)
    assert resp.status_code == 403
    assert "outside permitted workstation roots" in resp.json()["detail"]


def test_compress_multipart_upload(synthetic_wav):
    with open(synthetic_wav, "rb") as f:
        files = {"file": ("uploaded_tone.wav", f, "audio/wav")}
        data = {
            "speed": "4.0",
            "preset": "natural",
            "uniform": "false",
        }
        resp = client.post("/api/v1/compress", files=files, data=data)
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["speed"] == 4.0
    assert res["preset"] == "natural"
    assert Path(res["output_path"]).is_file()


def test_audio_range_streaming(synthetic_wav):
    # First compress to ensure file exists in OUTPUT_DIR
    payload = {"input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast"}
    comp_res = client.post("/api/v1/compress/json", json=payload).json()
    audio_url = comp_res["audio_url"]

    # Regular GET
    full_resp = client.get(audio_url)
    assert full_resp.status_code == 200
    assert full_resp.headers["accept-ranges"] == "bytes"

    # Partial Content Range Request (first 100 bytes)
    range_resp = client.get(audio_url, headers={"Range": "bytes=0-99"})
    assert range_resp.status_code == 206
    assert range_resp.headers["content-range"].startswith("bytes 0-99/")
    assert len(range_resp.content) == 100


def test_compare_and_reveal_keys(synthetic_wav):
    data = {
        "input_path": str(synthetic_wav),
        "speeds_str": "4.0,5.0",
    }
    resp = client.post("/api/v1/compare", data=data)
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert "folder_id" in res
    assert len(res["tracks"]) > 0

    folder_id = res["folder_id"]

    # Test listing comparisons
    list_resp = client.get("/api/v1/comparisons")
    assert list_resp.status_code == 200
    folders = [f["folder_id"] for f in list_resp.json()]
    assert folder_id in folders

    # Test key reveal
    key_resp = client.get(f"/api/v1/comparisons/{folder_id}/key")
    assert key_resp.status_code == 200
    key_data = key_resp.json()
    assert "track_01.wav" in key_data
    assert "speed" in key_data["track_01.wav"]


def test_heartbeat_and_disconnect():
    session_id = "test_session_xyz"
    hb = client.post("/api/v1/heartbeat", json={"session_id": session_id})
    assert hb.status_code == 200
    assert hb.json()["active_clients"] >= 1

    disc = client.post("/api/v1/client-disconnect", json={"session_id": session_id})
    assert disc.status_code == 200


def test_compress_mp3_format(synthetic_wav):
    payload = {
        "input_path": str(synthetic_wav),
        "speed": 5.0,
        "preset": "fast",
        "format": "mp3",
    }
    resp = client.post("/api/v1/compress/json", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["filename"].endswith(".mp3")
    assert data["format"] == "mp3"
    assert Path(data["output_path"]).is_file()

    # Verify audio streaming returns audio/mpeg
    audio_resp = client.get(data["audio_url"])
    assert audio_resp.status_code in (200, 206)
    assert "audio/mpeg" in audio_resp.headers["content-type"]



# --------------------------------------------------------------------------- Regression tests

def test_audio_endpoint_rejects_traversal_outside_output_dir():
    """ALLOWED_ROOTS spans most of D:, so the audio route must validate against
    OUTPUT_DIR instead. Percent-encoded dots are the interesting case: HTTP clients
    collapse a literal '..' before it ever reaches the server."""
    for path in (
        "/api/v1/audio/%2e%2e%2f%2e%2e%2fWorkspace%2fspeedman%2fCLAUDE.md",
        "/api/v1/audio/%2e%2e/%2e%2e/AI/Memory/global/rules.md",
        "/api/v1/audio/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ):
        resp = client.get(path)
        assert resp.status_code == 404, f"{path} leaked a file outside OUTPUT_DIR"


def test_transcribe_rejects_traversal_in_the_filename():
    """Since transcription moved to the source, the filename is only ever parsed for a
    stem and globbed inside SOURCE_SEARCH_DIRS -- it can no longer name a file to read.
    A traversal attempt now fails as 'cannot derive a source' rather than 404."""
    resp = client.post(
        "/api/v1/transcribe/%2e%2e%2f%2e%2e%2fWorkspace%2fspeedman%2fCLAUDE.md",
        json={"engine": "parakeet"},
    )
    assert resp.status_code == 400
    assert "CLAUDE" not in resp.text or "source_path" in resp.json()["detail"]


def test_transcribe_rejects_traversal_in_an_explicit_source_path():
    resp = client.post(
        "/api/v1/transcribe/x_5x_fast.wav",
        json={"engine": "parakeet",
              "source_path": "/mnt/d/Audio/Speed/../../../etc/passwd"},
    )
    assert resp.status_code == 403


def test_cors_is_not_wildcard():
    """A wildcard lets any page the user visits drive this loopback service."""
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert resp.headers.get("access-control-allow-origin") is None

    resp = client.get("/health", headers={"Origin": "http://127.0.0.1:8081"})
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:8081"


def test_unsupported_format_is_rejected_not_silently_downgraded(synthetic_wav):
    """The old code wrote a .wav but echoed back the format the caller asked for."""
    resp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast", "format": "ogg",
    })
    assert resp.status_code == 400
    assert "ogg" in resp.json()["detail"]

    resp = client.post("/api/v1/jobs/compress", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast", "format": "bogus",
    })
    assert resp.status_code == 400


def test_queued_job_rejects_unknown_preset(synthetic_wav):
    resp = client.post("/api/v1/jobs/compress", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "no_such_preset",
    })
    assert resp.status_code == 400


def test_compress_flac_format(synthetic_wav):
    resp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast", "format": "flac",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "flac"
    assert data["filename"].endswith(".flac")
    assert Path(data["output_path"]).is_file()
    assert client.get(data["audio_url"]).status_code == 200


def test_suffix_range_returns_tail_of_file(synthetic_wav):
    """'bytes=-500' means the LAST 500 bytes; the old parser read bytes 0-500."""
    comp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast",
    }).json()
    url = comp["audio_url"]
    size = int(client.get(url).headers["content-length"])

    resp = client.get(url, headers={"Range": "bytes=-500"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes {size - 500}-{size - 1}/{size}"
    assert len(resp.content) == 500


def test_range_end_past_eof_is_clamped(synthetic_wav):
    comp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast",
    }).json()
    url = comp["audio_url"]
    size = int(client.get(url).headers["content-length"])

    resp = client.get(url, headers={"Range": f"bytes=0-{size + 99999}"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-{size - 1}/{size}"

    # A start beyond EOF is still unsatisfiable.
    assert client.get(url, headers={"Range": f"bytes={size + 10}-"}).status_code == 416


def test_queued_job_result_carries_windows_output_path(synthetic_wav):
    """The Studio renders this field directly; without it queued jobs showed /mnt/d/..."""
    import time

    resp = client.post("/api/v1/jobs/compress", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast", "format": "wav",
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    for _ in range(200):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.1)

    assert job["status"] == "completed", job.get("error")
    result = job["result"]
    assert result["windows_output_path"].startswith("D:\\")
    assert result["format"] == "wav"


# --------------------------------------------------------------------------- Transcription direction

def test_compression_result_carries_its_source(synthetic_wav):
    """Provenance is what lets transcription find the original. Without it the only
    option is guessing from the filename."""
    resp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast",
    })
    data = resp.json()
    assert data["source_path"] == str(synthetic_wav)
    assert data["windows_source_path"]


def test_output_names_are_recognised_as_outputs():
    from app.api import looks_like_speedman_output, OUTPUT_DIR

    assert looks_like_speedman_output(OUTPUT_DIR / "talk_5x_fast.wav")
    assert looks_like_speedman_output(OUTPUT_DIR / "talk_6x_max_uniform.mp3")
    # A plain source file is not an output, wherever it sits.
    assert not looks_like_speedman_output(OUTPUT_DIR / "talk.wav")
    assert not looks_like_speedman_output(Path("/mnt/d/Output/Audio/talk_5x_fast.wav"))


def test_transcribe_refuses_to_run_on_a_speedman_output(synthetic_wav):
    """The whole point of the fix: parakeet returns nonsense on compressed audio, so
    asking it to is an error rather than a useless success."""
    comp = client.post("/api/v1/compress/json", json={
        "input_path": str(synthetic_wav), "speed": 5.0, "preset": "fast",
    }).json()

    resp = client.post(f"/api/v1/transcribe/{comp['filename']}",
                       json={"engine": "parakeet", "source_path": comp["output_path"]})
    assert resp.status_code == 400
    assert "nonsense" in resp.json()["detail"].lower()


def test_transcribe_rejects_a_source_outside_permitted_roots():
    resp = client.post("/api/v1/transcribe/anything.wav",
                       json={"engine": "parakeet", "source_path": "/etc/passwd"})
    assert resp.status_code == 403


def test_transcribe_reports_clearly_when_the_source_cannot_be_derived():
    resp = client.post("/api/v1/transcribe/not_an_output_name.wav", json={"engine": "parakeet"})
    assert resp.status_code == 400
    assert "source_path" in resp.json()["detail"]


def test_transcribe_404s_when_the_named_source_is_missing():
    resp = client.post("/api/v1/transcribe/x_5x_fast.wav",
                       json={"engine": "parakeet",
                             "source_path": "/mnt/d/Audio/Speed/definitely_missing_9f2.wav"})
    assert resp.status_code == 404


def test_find_source_for_output_recovers_the_stem(tmp_path, monkeypatch):
    from app import api as api_mod

    source = tmp_path / "episode.mp3"
    source.write_bytes(b"x")
    monkeypatch.setattr(api_mod, "SOURCE_SEARCH_DIRS", [tmp_path])

    assert api_mod.find_source_for_output("episode_5x_fast.wav") == source
    assert api_mod.find_source_for_output("episode_6x_max_uniform.flac") == source
    assert api_mod.find_source_for_output("unrelated.wav") is None


def test_sync_route_is_not_swallowed_by_the_filename_route():
    """/api/v1/transcribe/{filename:path} is greedy and declared after /sync. If the order
    ever flips, 'sync' is read as a filename and the endpoint silently performs a plain
    transcription instead -- returning 200 with none of the sync fields, which is exactly
    how this was first missed."""
    resp = client.post("/api/v1/transcribe/sync", json={"output_name": "no_such_output.wav"})
    # Reaching the sync handler means a 4xx about the missing transcript or map, never a
    # transcription result.
    assert resp.status_code in (400, 404)
    assert "media_api_response" not in resp.json()


def test_sync_requires_a_locatable_transcript():
    resp = client.post("/api/v1/transcribe/sync", json={"output_name": "mystery.wav"})
    assert resp.status_code == 400
    assert "transcript_path" in resp.json()["detail"]


def test_sync_rejects_a_transcript_outside_permitted_roots():
    resp = client.post("/api/v1/transcribe/sync", json={
        "output_name": "x_5x_fast.wav", "transcript_path": "/etc/passwd",
    })
    assert resp.status_code == 403


def test_transcribe_status_endpoint(monkeypatch):
    monkeypatch.setattr("app.api.check_media_api_online", lambda: True)
    monkeypatch.setattr(
        "app.media_api.get_media_api_job_status",
        lambda jid: {"job_id": jid, "status": "running", "progress_percent": 45.0, "speed": "Transcribing..."},
    )
    resp = client.get("/api/v1/transcribe/status/job-12345")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-12345"
    assert data["progress_percent"] == 45.0


def test_transcribe_status_endpoint_offline(monkeypatch):
    monkeypatch.setattr("app.api.check_media_api_online", lambda: False)
    resp = client.get("/api/v1/transcribe/status/job-12345")
    assert resp.status_code == 503


def test_sync_finds_transcript_in_media_api_dir(tmp_path, monkeypatch):
    from app import timemap_store, media_api

    out_name = "test_media_search_output_2x_fast.mp3"
    stem = "test_media_search_output_2x_fast"
    timemap_store.save(out_name, None, speed=2.0, sample_rate=24000)

    # Fake media_api audio dir in tmp_path
    fake_media_audio = tmp_path / "media_audio"
    fake_media_audio.mkdir()
    monkeypatch.setattr(media_api, "MEDIA_API_AUDIO_DIR", fake_media_audio)

    # Transcript is placed in media_api audio dir with stem "my_source_video"
    transcript_file = fake_media_audio / "my_source_video.transcript.json"
    transcript_file.write_text(json.dumps({
        "segments": [{"start": 0.0, "end": 2.0, "text": "Testing media api dir discovery."}]
    }), encoding="utf-8")

    # Source was a video elsewhere
    fake_video = tmp_path / "my_source_video.mp4"
    fake_video.touch()

    try:
        resp = client.post("/api/v1/transcribe/sync", json={
            "output_name": out_name,
            "source_path": str(fake_video),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "synced"
        assert "source_vtt_url" in data
        assert "vtt_url" in data
        assert "srt_url" in data
        assert "source_srt_url" in data
        assert "txt_url" in data
    finally:
        for ext in (".vtt", ".source.vtt", ".srt", ".source.srt", ".txt", ".synced.json", ".timemap.npz"):
            p = OUTPUT_DIR / f"{stem}{ext}"
            if p.exists():
                p.unlink()


def test_get_synced_transcript_export_formats():
    from app import timemap_store

    out_name = "test_export_formats_3x_fast.mp3"
    stem = "test_export_formats_3x_fast"
    timemap_store.save(out_name, None, speed=3.0, sample_rate=24000)

    synced_json = OUTPUT_DIR / f"{stem}.synced.json"
    synced_json.write_text(json.dumps({
        "output": out_name,
        "speed": 3.0,
        "source_duration_s": 10.0,
        "output_duration_s": 3.33,
        "segments": [
            {
                "start": 0.5,
                "end": 1.5,
                "source_start": 1.5,
                "source_end": 4.5,
                "text": "First segment here.",
                "words": []
            },
            {
                "start": 2.0,
                "end": 3.0,
                "source_start": 6.0,
                "source_end": 9.0,
                "text": "Second segment follows.",
                "words": []
            }
        ]
    }), encoding="utf-8")

    try:
        # Default JSON response with export links
        resp = client.get(f"/api/v1/transcript/{out_name}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["srt_url"] == f"/api/v1/audio/{stem}.srt"
        assert data["source_srt_url"] == f"/api/v1/audio/{stem}.source.srt"
        assert data["txt_url"] == f"/api/v1/audio/{stem}.txt"
        assert (OUTPUT_DIR / f"{stem}.srt").is_file()
        assert (OUTPUT_DIR / f"{stem}.txt").is_file()

        # Direct audio stream of srt and txt
        resp_srt = client.get(f"/api/v1/audio/{stem}.srt")
        assert resp_srt.status_code == 200
        assert "text/plain" in resp_srt.headers.get("content-type", "")

        resp_txt = client.get(f"/api/v1/audio/{stem}.txt")
        assert resp_txt.status_code == 200
        assert "text/plain" in resp_txt.headers.get("content-type", "")

        # Formatted export downloads
        srt_exp = client.get(f"/api/v1/transcript/{out_name}?format=srt")
        assert srt_exp.status_code == 200
        assert "-->" in srt_exp.text
        assert "First segment here." in srt_exp.text
        assert "attachment" in srt_exp.headers.get("content-disposition", "")

        srt_src_exp = client.get(f"/api/v1/transcript/{out_name}?format=srt&source=true")
        assert srt_src_exp.status_code == 200
        assert "00:00:01,500 --> 00:00:04,500" in srt_src_exp.text

        vtt_exp = client.get(f"/api/v1/transcript/{out_name}?format=vtt")
        assert vtt_exp.status_code == 200
        assert "WEBVTT" in vtt_exp.text

        txt_exp = client.get(f"/api/v1/transcript/{out_name}?format=txt")
        assert txt_exp.status_code == 200
        assert "First segment here." in txt_exp.text

        txt_ts_exp = client.get(f"/api/v1/transcript/{out_name}?format=txt&timestamps=true")
        assert txt_ts_exp.status_code == 200
        assert "[" in txt_ts_exp.text
    finally:
        for ext in (".vtt", ".source.vtt", ".srt", ".source.srt", ".txt", ".synced.json", ".timemap.npz"):
            p = OUTPUT_DIR / f"{stem}{ext}"
            if p.exists():
                p.unlink()


# --------------------------------------------------------------------------- client sessions

def test_session_window_tolerates_a_throttled_background_tab():
    """Browsers throttle setInterval to about once a minute in a hidden tab. The window
    must exceed that, or an open-but-backgrounded window is forgotten and the watchdog
    shuts the service down underneath it -- which is exactly what used to happen at 12s."""
    from app.api import SessionManager

    mgr = SessionManager()
    assert mgr.session_ttl > 60.0, "must outlast a once-per-minute throttled heartbeat"


def test_a_heartbeat_keeps_a_client_counted_well_past_the_old_window():
    import time as _time
    from app.api import SessionManager

    mgr = SessionManager()
    mgr.sessions["tab"] = _time.time() - 30.0   # silent for 30s, as a hidden tab would be
    assert mgr.active_client_count() == 1


def test_a_long_silent_client_is_eventually_dropped():
    import time as _time
    from app.api import SessionManager

    mgr = SessionManager()
    mgr.sessions["gone"] = _time.time() - (mgr.session_ttl + 5)
    assert mgr.active_client_count() == 0


def test_explicit_disconnect_does_not_wait_for_the_window():
    """Closing a tab sends a beacon, so a clean close still releases RAM promptly."""
    client.post("/api/v1/heartbeat", json={"session_id": "closing_tab"})
    before = client.get("/health").json()["active_clients"]
    client.post("/api/v1/client-disconnect", json={"session_id": "closing_tab"})
    assert client.get("/health").json()["active_clients"] == before - 1


def test_cors_allows_loopback_and_file_origins_but_rejects_external():
    # Loopback ports
    for origin in ("http://127.0.0.1:8081", "http://localhost:8081", "http://localhost:8080", "null"):
        resp = client.options(
            "/api/v1/ingest/url",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200, f"Failed preflight for origin {origin}"
        assert resp.headers.get("access-control-allow-origin") == origin

    # External origin rejected
    evil = "https://evil.com"
    resp = client.options(
        "/api/v1/ingest/url",
        headers={
            "Origin": evil,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") != evil


def test_open_workstation_folder(monkeypatch):
    from unittest.mock import MagicMock
    import subprocess
    mock_popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    resp = client.post("/api/v1/tools/open-folder", json={"folder": "audio"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert "Speed" in resp.json()["path"]
    mock_popen.assert_called_once()

    bad = client.post("/api/v1/tools/open-folder", json={"folder": "non_existent"})
    assert bad.status_code == 400


def test_list_saved_outputs(tmp_path, monkeypatch):
    from app import api
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)

    f1 = tmp_path / "podcast_5x_fast.mp3"
    f1.write_bytes(b"fake audio data")
    f1_json = tmp_path / "podcast_5x_fast.synced.json"
    f1_json.write_text('{"segments": []}', encoding="utf-8")

    f2 = tmp_path / "talk_3.5x_fast.wav"
    f2.write_bytes(b"fake wav data")

    resp = client.get("/api/v1/outputs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {item["filename"]: item for item in data}
    assert "podcast_5x_fast.mp3" in names
    assert names["podcast_5x_fast.mp3"]["has_transcript"] is True
    assert names["podcast_5x_fast.mp3"]["speed"] == 5.0
    assert "talk_3.5x_fast.wav" in names
    assert names["talk_3.5x_fast.wav"]["has_transcript"] is False
    assert names["talk_3.5x_fast.wav"]["speed"] == 3.5


def test_speed_reader_ui_elements():
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Check Google Fonts
    assert "Lexend+Deca" in html
    assert "Atkinson+Hyperlegible" in html

    # Check WPM dashboard elements
    assert 'id="wpmInstantVal"' in html
    assert 'id="wpmSparklineSvg"' in html
    assert 'id="wpmTierBadge"' in html
    assert 'id="wpmTimeSavedVal"' in html

    # Check Tri-Focus RSVP elements
    assert 'id="rsvpCard"' in html
    assert 'id="rsvpORP"' in html
    assert 'id="rsvpPrevWord"' in html
    assert 'id="rsvpNextWord"' in html

    # Check Speed Reader settings drawer & presets
    assert 'id="readerSettingsDrawer"' in html
    assert 'id="rsAutoScrollMode"' in html
    assert 'id="rsBionicStrength"' in html
    assert 'id="rsThemeSelect"' in html
    assert 'id="playerRateButtons"' in html

    # Check Saved library modal
    assert 'id="savedLibraryModal"' in html
    assert 'id="savedLibraryList"' in html
    assert 'openSavedOutputsModal' in html


def test_safe_stem_handles_decimal_speeds_and_dots():
    from app.paths import safe_stem

    assert safe_stem("talk_5.5x_fast.mp3") == "talk_5.5x_fast"
    assert safe_stem("talk_5.5x_fast") == "talk_5.5x_fast"
    assert safe_stem("Dr. Smith - Lecture 01_5.5x_fast.synced.json") == "Dr. Smith - Lecture 01_5.5x_fast"
    assert safe_stem("Dr. Smith - Lecture 01_5.5x_fast.vtt") == "Dr. Smith - Lecture 01_5.5x_fast"
    assert safe_stem("Dr. Smith - Lecture 01_5.5x_fast.mp3") == "Dr. Smith - Lecture 01_5.5x_fast"
    assert safe_stem("Dr. Smith - Lecture 01_5.5x_fast") == "Dr. Smith - Lecture 01_5.5x_fast"
    assert safe_stem("/mnt/d/Audio/Speed/interview_3.25x_fast.wav") == "interview_3.25x_fast"
    assert safe_stem("D:\\Audio\\Speed\\interview_3.25x_fast.wav") == "interview_3.25x_fast"


def test_get_synced_transcript_with_decimal_speeds(tmp_path, monkeypatch):
    from app import api
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)

    base = "interview_5.5x_fast"
    json_file = tmp_path / f"{base}.synced.json"
    json_file.write_text(json.dumps({
        "output": f"{base}.mp3",
        "speed": 5.5,
        "segments": [{"text": "Testing decimal speed transcripts"}],
    }), encoding="utf-8")

    # 1. With .mp3
    resp1 = client.get(f"/api/v1/transcript/{base}.mp3")
    assert resp1.status_code == 200
    assert resp1.json()["speed"] == 5.5
    assert len(resp1.json()["segments"]) == 1

    # 2. Bare stem without extension
    resp2 = client.get(f"/api/v1/transcript/{base}")
    assert resp2.status_code == 200
    assert resp2.json()["speed"] == 5.5

    # 3. With .synced.json
    resp3 = client.get(f"/api/v1/transcript/{base}.synced.json")
    assert resp3.status_code == 200
    assert resp3.json()["speed"] == 5.5


def test_timemap_store_roundtrip_with_decimal_speeds(tmp_path, monkeypatch):
    from app import timemap_store
    monkeypatch.setattr(timemap_store, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(timemap_store, "TIMEMAP_DIR", tmp_path / ".timemaps")

    name_with_ext = "talk_5.5x_fast.mp3"
    name_bare = "talk_5.5x_fast"

    timemap_store.save(name_with_ext, None, speed=5.5, sample_rate=24000)

    # Both with extension and bare stem should load successfully
    loaded1 = timemap_store.load(name_with_ext)
    assert loaded1.speed == 5.5

    loaded2 = timemap_store.load(name_bare)
    assert loaded2.speed == 5.5


def test_audiophile_preset_in_api():
    resp = client.get("/api/v1/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "audiophile" in data
    assert data["audiophile"]["post"]["deess_db"] == 4.0
    assert data["audiophile"]["post"]["highpass_hz"] == 80.0
    assert data["audiophile"]["post"]["warmth_db"] == 2.0


def test_transcribe_pipeline_offline(monkeypatch, synthetic_wav):
    monkeypatch.setattr("app.api.check_media_api_online", lambda: False)
    resp = client.post("/api/v1/transcribe/pipeline", json={
        "output_name": "test_5x_fast.wav",
        "source_path": str(synthetic_wav),
        "wait": False,
    })
    assert resp.status_code == 503
    assert "offline" in resp.json()["detail"].lower()


def test_transcribe_pipeline_nowait(monkeypatch, synthetic_wav):
    monkeypatch.setattr("app.api.check_media_api_online", lambda: True)
    monkeypatch.setattr(
        "app.api.send_to_media_api_transcribe",
        lambda src, engine="parakeet": {"job_id": "job-test-nowait", "status": "queued"},
    )
    resp = client.post("/api/v1/transcribe/pipeline", json={
        "output_name": "test_5x_fast.wav",
        "source_path": str(synthetic_wav),
        "wait": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "forwarded_to_media_api"
    assert data["job_id"] == "job-test-nowait"


def test_transcribe_pipeline_wait_flow(tmp_path, monkeypatch, synthetic_wav):
    from app import timemap_store, obsidian
    monkeypatch.setattr("app.api.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(timemap_store, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(timemap_store, "TIMEMAP_DIR", tmp_path / ".timemaps")
    monkeypatch.setattr(obsidian, "NOTEBOOK_ROOT", tmp_path / "1Notebook")
    (tmp_path / "1Notebook" / "Summaries").mkdir(parents=True, exist_ok=True)

    out_name = "pipeline_test_5x_fast.wav"
    timemap_store.save(out_name, None, speed=5.0, sample_rate=24000)

    # Transcript json created by mock ASR
    t_file = tmp_path / "synthetic.transcript.json"
    t_file.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "First segment about technology."},
            {"start": 2.0, "end": 4.0, "text": "Second segment discussing future architectures."}
        ]
    }), encoding="utf-8")

    monkeypatch.setattr("app.api.check_media_api_online", lambda: True)
    monkeypatch.setattr(
        "app.api.send_to_media_api_transcribe",
        lambda src, engine="parakeet": {"job_id": "job-pipeline-full", "status": "queued"},
    )
    monkeypatch.setattr(
        "app.media_api.get_media_api_job_status",
        lambda jid: {
            "job_id": jid,
            "status": "completed",
            "output_files": {"transcript_json": str(t_file)},
        },
    )

    resp = client.post("/api/v1/transcribe/pipeline", json={
        "output_name": out_name,
        "source_path": str(synthetic_wav),
        "wait": True,
        "auto_chapters": True,
        "auto_obsidian": True,
        "use_llm": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["job_id"] == "job-pipeline-full"
    assert data["sync"]["status"] == "synced"
    assert data["chapters"]["count"] >= 1
    assert data["obsidian_note"]["status"] == "success"




