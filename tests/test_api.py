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

