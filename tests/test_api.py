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
