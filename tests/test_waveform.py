import numpy as np
import pytest
from fastapi.testclient import TestClient

from speedman.post import compute_waveform_peaks
from app.api import app, OUTPUT_DIR
from speedman import io as sio
import json


def test_compute_waveform_peaks_empty():
    peaks = compute_waveform_peaks(np.array([], dtype=np.float32), num_bins=10)
    assert len(peaks) == 10
    assert all(p == 0.0 for p in peaks)


def test_compute_waveform_peaks_normalization():
    # Signal with known max amplitude 0.5
    y = np.linspace(-0.5, 0.5, 1200, dtype=np.float32)
    peaks = compute_waveform_peaks(y, num_bins=12)
    assert len(peaks) == 12
    # The max peak should be normalized to 1.0
    assert max(peaks) == pytest.approx(1.0, abs=0.05)
    assert min(peaks) >= 0.0


def test_waveform_api_endpoint(tmp_path, monkeypatch):
    client = TestClient(app)

    # Create dummy audio file in OUTPUT_DIR
    sr = 24000
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
    filename = "test_waveform_clip.wav"
    out_path = OUTPUT_DIR / filename
    sio.save(out_path, y, sr)

    try:
        res = client.get(f"/api/v1/waveform/{filename}")
        assert res.status_code == 200
        data = res.json()
        assert "peaks" in data
        assert len(data["peaks"]) == 120
        assert max(data["peaks"]) == pytest.approx(1.0, abs=0.05)

        # Verify cached peaks file was created
        peaks_file = OUTPUT_DIR / "test_waveform_clip.peaks.json"
        assert peaks_file.is_file()
    finally:
        if out_path.exists():
            out_path.unlink()
        peaks_file = OUTPUT_DIR / "test_waveform_clip.peaks.json"
        if peaks_file.exists():
            peaks_file.unlink()


def test_waveform_endpoint_rejects_traversal_and_missing_files():
    client = TestClient(app)
    # Traversal should 400 or 404
    res = client.get("/api/v1/waveform/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
    assert res.status_code in (400, 404)

    # Dot-dot bare should 400 or 404
    res_dotdot = client.get("/api/v1/waveform/%2e%2e")
    assert res_dotdot.status_code in (400, 404)

    # Non-existent file should 404
    res2 = client.get("/api/v1/waveform/non_existent_audio_999.wav")
    assert res2.status_code == 404


def test_compute_waveform_peaks_edge_cases():
    # 1. NaN and Inf handling
    y_nan = np.array([np.nan, np.inf, -np.inf, 0.5, 1.0], dtype=np.float32)
    peaks = compute_waveform_peaks(y_nan, num_bins=5)
    assert len(peaks) == 5
    assert not any(np.isnan(p) for p in peaks)
    assert not any(np.isinf(p) for p in peaks)
    assert max(peaks) == 1.0
    # Ensure JSON serializable without NaN
    dumped = json.dumps({"peaks": peaks})
    assert "NaN" not in dumped

    # 2. Short clip (< num_bins) interpolation
    y_short = np.array([0.2, 0.8, 0.4], dtype=np.float32)
    peaks_short = compute_waveform_peaks(y_short, num_bins=6)
    assert len(peaks_short) == 6
    assert max(peaks_short) == 1.0
    assert min(peaks_short) >= 0.0

    # 3. Scalar/1-sample array
    peaks_scalar = compute_waveform_peaks(np.array([0.5]), num_bins=3)
    assert len(peaks_scalar) == 3
    assert all(p == 1.0 for p in peaks_scalar)

    # 4. Multi-channel input
    y_stereo = np.zeros((100, 2), dtype=np.float32)
    peaks_stereo = compute_waveform_peaks(y_stereo, num_bins=10)
    assert len(peaks_stereo) == 10

    # 5. num_bins <= 0
    peaks_zero = compute_waveform_peaks(np.array([0.5, 0.2]), num_bins=0)
    assert len(peaks_zero) == 1
