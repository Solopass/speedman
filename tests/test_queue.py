"""Tests for async job queue and process cancellation."""
import time
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.api import app
from app.queue import JobManager

client = TestClient(app)


@pytest.fixture
def sample_wav(tmp_path):
    wav = tmp_path / "sample_queue.wav"
    sr = 24000
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    y = np.sin(2 * np.pi * 440 * t)
    y[sr // 2 : sr] = 0.0
    sf.write(str(wav), y, sr)
    return wav


def test_job_queue_lifecycle(sample_wav):
    mgr = JobManager(max_history=10)
    job = mgr.submit(input_path=sample_wav, speed=4.0, preset="fast", output_dir=sample_wav.parent)
    assert job.status in ("queued", "processing")
    assert job.job_id.startswith("job_")

    # Wait for completion (2s audio at 4x finishes in < 1s)
    for _ in range(50):
        if job.status == "completed":
            break
        time.sleep(0.1)

    assert job.status == "completed"
    assert job.result is not None
    assert job.result["status"] == "success"
    assert job.progress.progress_pct == 100.0


def test_job_queue_cancellation(sample_wav):
    mgr = JobManager(max_history=10)
    job = mgr.submit(input_path=sample_wav, speed=4.0, preset="fast", output_dir=sample_wav.parent)
    # Cancel immediately
    mgr.cancel_job(job.job_id)
    assert job.is_cancelled() is True

    for _ in range(30):
        if job.status == "cancelled":
            break
        time.sleep(0.1)

    assert job.status == "cancelled"


def test_api_jobs_endpoints(sample_wav):
    # Submit via API
    resp = client.post("/api/v1/jobs/compress", json={
        "input_path": str(sample_wav),
        "speed": 5.0,
        "preset": "fast",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    job_id = data["job_id"]

    # Poll status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ("queued", "processing", "completed")

    # List jobs
    list_resp = client.get("/api/v1/jobs")
    assert list_resp.status_code == 200
    jobs = list_resp.json()
    assert any(j["job_id"] == job_id for j in jobs)

    # Test cancel endpoint on non-existent job -> 404
    bad_cancel = client.post("/api/v1/jobs/nonexistent_xyz/cancel")
    assert bad_cancel.status_code == 404


# --------------------------------------------------------------------------- URL-sourced jobs

def test_job_requires_an_input_path_or_a_source_url():
    from app.queue import SpeedmanJob
    import pytest as _pytest

    with _pytest.raises(ValueError, match="input_path or a source_url"):
        SpeedmanJob(job_id="j", input_path=None, source_url=None)


def test_url_job_carries_the_url_and_has_no_filename_yet():
    """input_path is resolved by the worker after the download, so to_dict has to cope
    with it being absent -- the UI polls this before the download finishes."""
    from app.queue import SpeedmanJob

    job = SpeedmanJob(job_id="j", source_url="https://example.com/v")
    assert job.input_path is None
    d = job.to_dict()
    assert d["source_url"] == "https://example.com/v"
    assert d["input_filename"] is None
