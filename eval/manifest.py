"""The fixed clip set.

Fixed on purpose: a moving clip set makes two runs incomparable, which defeats the
point of having an eval. Offsets are chosen to land in continuous speech, past intros
and music beds. Add clips freely; changing an existing clip's offset or duration
invalidates every result previously recorded against it, so prefer adding a new entry.

Sources live outside the repo (they are hours of copyrighted podcast audio). A clip
whose source is missing is skipped with a note rather than failing the run, so the
harness still works on a machine that has only some of them.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MEDIA_ROOT = Path("/mnt/d/Output/Audio")

# Bump when `Clip.load` changes in a way that alters the decoded samples. It feeds the
# fingerprint, so a change invalidates cached results instead of silently mixing audio
# decoded two different ways into one comparison.
LOADER_VERSION = 2


@dataclass(frozen=True)
class Clip:
    label: str
    source: Path
    offset_s: float
    duration_s: float
    kind: str = "conversational"  # affects silence fraction, so worth grouping by

    @property
    def available(self) -> bool:
        return self.source.is_file()

    def fingerprint(self) -> str:
        """Identifies the exact audio span, so cached results cannot silently
        outlive a manifest edit."""
        key = f"{self.source.name}|{self.offset_s}|{self.duration_s}|v{LOADER_VERSION}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def load(self, sr: int) -> np.ndarray:
        """Decode just this clip's span to mono float32 at `sr`.

        `speedman.io.load` decodes the whole file, which for the 9-hour audiobook means
        ~90 seconds and gigabytes of RAM to obtain 90 seconds of audio. Seeking before
        -i makes ffmpeg jump straight to the span.
        """
        if not self.available:
            raise FileNotFoundError(self.source)
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{self.offset_s:.3f}",
            "-t", f"{self.duration_s:.3f}",
            "-i", str(self.source),
            "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed extracting {self.label} from {self.source.name}:\n"
                f"{proc.stderr.decode()[:400]}"
            )
        y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
        want = int(self.duration_s * sr)
        if len(y) < want * 0.9:
            raise RuntimeError(
                f"{self.label}: got {len(y)/sr:.1f}s but expected {self.duration_s:.1f}s "
                f"-- is the source shorter than offset {self.offset_s:.0f}s?"
            )
        return y[:want]


_DATING = MEDIA_ROOT / "Autistic_Dating_Differences_Explained_k11VxPAgtiw.mp3"          # 2136s
_BEHAVIORS = MEDIA_ROOT / "Autistic_Behaviors_People_Harshly_Judge_Without_Realizing_yQeW-U5F4BA.mp3"  # 2117s
_AUDIOBOOK = MEDIA_ROOT / "THE_FOURTH_DIMENSION_-_FULL_AudioBook_Greatest_AudioBooks_vuwOgKYwprQ.mp3"  # 32706s

# Four non-overlapping excerpts per source. Between-clip variance is the dominant
# error term -- with three clips the standard error on a sweep was large enough to
# swamp a real difference (signal/noise 1.4x at 2.5x), so n matters more here than
# clip length does.
CLIPS: list[Clip] = [
    Clip("podcast_dating", _DATING, 300.0, 90.0, "conversational"),
    Clip("podcast_dating_b", _DATING, 700.0, 90.0, "conversational"),
    Clip("podcast_dating_c", _DATING, 1200.0, 90.0, "conversational"),
    Clip("podcast_dating_d", _DATING, 1700.0, 90.0, "conversational"),

    Clip("podcast_behaviors", _BEHAVIORS, 300.0, 90.0, "conversational"),
    Clip("podcast_behaviors_b", _BEHAVIORS, 700.0, 90.0, "conversational"),
    Clip("podcast_behaviors_c", _BEHAVIORS, 1200.0, 90.0, "conversational"),
    Clip("podcast_behaviors_d", _BEHAVIORS, 1700.0, 90.0, "conversational"),

    Clip("audiobook_4d", _AUDIOBOOK, 600.0, 90.0, "narration"),
    Clip("audiobook_4d_b", _AUDIOBOOK, 3600.0, 90.0, "narration"),
    Clip("audiobook_4d_c", _AUDIOBOOK, 9000.0, 90.0, "narration"),
    Clip("audiobook_4d_d", _AUDIOBOOK, 18000.0, 90.0, "narration"),
]


def by_label(labels: list[str] | None = None) -> list[Clip]:
    """Select clips by label; None means all of them."""
    if not labels:
        return list(CLIPS)
    index = {c.label: c for c in CLIPS}
    unknown = [l for l in labels if l not in index]
    if unknown:
        raise ValueError(f"unknown clip label(s) {unknown}; available: {sorted(index)}")
    return [index[l] for l in labels]


def available(clips: list[Clip]) -> tuple[list[Clip], list[Clip]]:
    """Split into (present, missing)."""
    return [c for c in clips if c.available], [c for c in clips if not c.available]
