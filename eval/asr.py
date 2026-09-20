"""ASR scoring backend: the parakeet engine already running behind Media API.

Why Media API rather than the `eval` extra in pyproject.toml: that extra pulls torch,
torchaudio and transformers (~2 GB) to run a wav2vec2 model, duplicating an ASR engine
this workstation already has warm on port 8080. `analyze.py` avoids torch for exactly
this reason; the eval harness should not reintroduce it.

The tradeoff to keep in mind when reading results: parakeet was not trained on
time-compressed speech, so its absolute WER on a 5x clip overstates how unintelligible
that clip is to a human. Differences BETWEEN conditions are the usable signal.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

MEDIA_API_BASE = os.getenv("MEDIA_API_URL", "http://127.0.0.1:8080")
POLL_INTERVAL_S = 1.5
DEFAULT_TIMEOUT_S = 600.0


class ASRUnavailable(RuntimeError):
    """Media API is not reachable, so nothing can be scored."""


class ASRFailed(RuntimeError):
    """Media API accepted the job but could not transcribe it."""


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str
    model: str
    n_segments: int

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        segments = data.get("segments") or []
        text = " ".join(str(s.get("text", "")).strip() for s in segments).strip()
        return cls(
            text=text,
            engine=str(data.get("engine", "unknown")),
            model=str(data.get("model", "unknown")),
            n_segments=len(segments),
        )


def is_online(timeout: float = 3.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.get(f"{MEDIA_API_BASE}/api/v1/health").status_code == 200
    except Exception:
        return False


def require_online() -> None:
    if not is_online():
        raise ASRUnavailable(
            f"Media API at {MEDIA_API_BASE} is not responding. Start it and re-run; "
            "the harness needs an ASR engine to score against."
        )


def transcribe(audio_path: Path, engine: str = "parakeet",
               timeout_s: float = DEFAULT_TIMEOUT_S) -> Transcript:
    """Transcribe one file, blocking until Media API reports a terminal status.

    Media API writes `<stem>.transcript.json` beside the audio, which is why the
    harness keeps its clips in their own directory rather than in D:\\Audio\\Speed.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{MEDIA_API_BASE}/api/v1/transcribe",
            json={"source": str(audio_path), "engine": engine},
        )
        if resp.status_code not in (200, 201, 202):
            raise ASRFailed(f"submit failed ({resp.status_code}): {resp.text[:300]}")
        job_id = resp.json().get("job_id")
        if not job_id:
            raise ASRFailed(f"no job_id in response: {resp.text[:300]}")

        deadline = time.monotonic() + timeout_s
        while True:
            if time.monotonic() > deadline:
                raise ASRFailed(f"transcription of {audio_path.name} timed out after {timeout_s:.0f}s")
            status = client.get(f"{MEDIA_API_BASE}/api/v1/status/{job_id}").json()
            state = str(status.get("status", "")).lower()
            if state == "completed":
                break
            if state in ("failed", "error", "cancelled"):
                raise ASRFailed(f"{audio_path.name}: {status.get('error') or state}")
            time.sleep(POLL_INTERVAL_S)

    return _read_result(status, audio_path)


def _read_result(status: dict, audio_path: Path) -> Transcript:
    outputs = status.get("output_files") or {}
    json_path = outputs.get("transcript_json")
    if json_path and Path(json_path).is_file():
        return Transcript.from_json(json.loads(Path(json_path).read_text(encoding="utf-8")))

    # Fall back to the sibling files Media API writes, then to the plain .txt.
    sibling = audio_path.with_suffix(".transcript.json")
    if sibling.is_file():
        return Transcript.from_json(json.loads(sibling.read_text(encoding="utf-8")))

    text_path = outputs.get("text") or str(audio_path.with_suffix(".txt"))
    if Path(text_path).is_file():
        return Transcript(
            text=Path(text_path).read_text(encoding="utf-8").strip(),
            engine="unknown", model="unknown", n_segments=0,
        )

    raise ASRFailed(f"transcription completed but no output found for {audio_path.name}")
