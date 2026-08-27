"""Audio load/save.

Units: everything in this package is mono float32 at `Config.sample_rate`.
Sample counts are ints; times are floats in seconds. Most DSP bugs are unit
bugs, so every public function states which it takes.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

WORKING_SR = 24000
_FFMPEG_EXTS = {".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".webm", ".flac", ".wma"}


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load(path: str | Path, sr: int = WORKING_SR) -> np.ndarray:
    """Load any audio file to mono float32 at `sr`. Returns shape (n,)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in _FFMPEG_EXTS or path.suffix.lower() not in {".wav", ".aiff"}:
        if not have_ffmpeg():
            raise RuntimeError(
                f"{path.suffix} needs ffmpeg, which was not found on PATH. "
                "Install ffmpeg, or convert the file to WAV first."
            )
        return _load_via_ffmpeg(path, sr)

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return resample(y, file_sr, sr)


def _load_via_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """Decode straight to the working format; ffmpeg resamples better than we would."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}:\n{proc.stderr.decode()[:500]}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


def save(path: str | Path, y: np.ndarray, sr: int = WORKING_SR) -> None:
    """Write wav directly; anything else goes out through ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)

    if path.suffix.lower() == ".wav":
        sf.write(str(path), y, sr, subtype="PCM_16")
        return

    if not have_ffmpeg():
        raise RuntimeError(f"writing {path.suffix} needs ffmpeg; use .wav instead")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "out.wav"
        sf.write(str(tmp), y, sr, subtype="PCM_16")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), str(path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {path.name}:\n{proc.stderr.decode()[:500]}")


def to_eval_rate(y: np.ndarray, sr: int = WORKING_SR) -> np.ndarray:
    """16 kHz for wav2vec2-family models. Easy to forget, silently wrong if missed."""
    return resample(y, sr, 16000)
