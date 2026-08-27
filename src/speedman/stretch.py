"""Time-stretch backends behind one interface.

IMPORTANT, and counter-intuitive enough that it is worth stating loudly:
only ONE backend can apply a time map.

  rubberband     constant rate + TIME MAP   <- the real engine
  phasevocoder   constant rate only         (librosa time_stretch is scalar-only)
  wsola          constant rate only         (audiotsm has no variable hop, and
                                             silently DROPS input above 2x)
  resample       constant rate only         (a varying rate warbles the pitch)

The other three are eval controls, not fallbacks. They RAISE on stretch_map
rather than silently approximating -- a quiet approximation here would look like
a working pipeline producing quietly wrong audio.
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def find_rubberband() -> str | None:
    """Locate the rubberband binary.

    Checked in order: the SPEEDMAN_RUBBERBAND environment variable, the system
    PATH, then a `bin/` folder beside the project or the working directory.

    The `bin/` fallback exists because editing PATH is by far the most
    error-prone step in setting this up on Windows, and dropping an .exe into a
    folder is not.
    """
    env = os.environ.get("SPEEDMAN_RUBBERBAND")
    if env and Path(env).exists():
        return env

    on_path = shutil.which("rubberband")
    if on_path:
        return on_path

    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for root in roots:
        for name in ("rubberband.exe", "rubberband"):
            candidate = root / "bin" / name
            if candidate.exists():
                return str(candidate)
    return None


class BackendUnavailable(RuntimeError):
    pass


def _safe(y: np.ndarray) -> np.ndarray:
    """pyrubberband round-trips through a PCM_16 temp WAV, so anything above
    full scale is silently hard-clipped before rubberband ever sees it."""
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y / peak * 0.98 if peak >= 1.0 else y


class TimeStretcher(ABC):
    name = "base"
    supports_time_map = False

    def available(self) -> bool:
        return True

    @abstractmethod
    def stretch(self, y: np.ndarray, sr: int, rate: float) -> np.ndarray:
        """Uniform stretch. rate > 1 makes the signal shorter."""

    def stretch_map(self, y: np.ndarray, sr: int, time_map) -> np.ndarray:
        raise NotImplementedError(
            f"backend {self.name!r} has no time-map path, so it cannot do non-uniform "
            "compression. Use --backend rubberband, or --uniform for a constant rate. "
            "(This is a hard limitation of the library, not a missing feature here.)"
        )

    def staged(self, y: np.ndarray, sr: int, rate: float, threshold: float = 4.0) -> np.ndarray:
        """Chain equal passes so each stays in a regime the algorithm handles.

        Frame size is a per-backend concern; where a backend exposes one it must
        scale it by the cumulative compression, or pass 2 spans several phonemes
        and smears everything pass 1 preserved.
        """
        if rate <= threshold:
            return self.stretch(y, sr, rate)
        n_passes = int(np.ceil(np.log(rate) / np.log(threshold)))
        per = rate ** (1.0 / n_passes)
        out = y
        for _ in range(n_passes):
            out = self.stretch(out, sr, per)
        return out


class ResampleStretcher(TimeStretcher):
    """Control / worst case: the chipmunk baseline. Changes pitch by design."""
    name = "resample"

    def stretch(self, y, sr, rate):
        n_out = int(round(len(y) / rate))
        return np.interp(np.linspace(0, len(y) - 1, n_out),
                         np.arange(len(y)), y).astype(np.float32)


class PhaseVocoderStretcher(TimeStretcher):
    """librosa phase vocoder -- the 'what any hacker would do' baseline."""
    name = "phasevocoder"

    def stretch(self, y, sr, rate):
        import librosa
        return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=rate)


class WSOLAStretcher(TimeStretcher):
    """audiotsm WSOLA.

    Configured explicitly rather than via set_speed: audiotsm's default puts
    analysis_hop > frame_length above ~2x, at which point it silently discards
    input (60% at 5x, 80% at 10x measured) -- which would delete exactly the
    consonants this project exists to protect.
    """
    name = "wsola"

    def available(self) -> bool:
        try:
            import audiotsm  # noqa: F401
            return True
        except ImportError:
            return False

    def stretch(self, y, sr, rate):
        try:
            from audiotsm import wsola
            from audiotsm.io.array import ArrayReader, ArrayWriter
        except ImportError as e:
            raise BackendUnavailable("audiotsm is not installed") from e
        frame = 2048
        analysis_hop = frame // 2
        synthesis_hop = max(1, int(round(analysis_hop / rate)))
        tsm = wsola(1, frame_length=frame, analysis_hop=analysis_hop,
                    synthesis_hop=synthesis_hop)
        reader = ArrayReader(np.asarray(y, dtype=np.float32)[None, :])
        writer = ArrayWriter(1)
        tsm.run(reader, writer)
        return writer.data.flatten().astype(np.float32)


class RubberBandStretcher(TimeStretcher):
    """The only backend with a real time-map path.

    Note `--formant` is deliberately NOT used: it only does anything when pitch
    shifting, and this pipeline never pitch-shifts. Formants survive because
    this is TSM rather than resampling.
    """
    name = "rubberband"
    supports_time_map = True

    def __init__(self, r3: bool = False):
        self.rbargs = {"-3": ""} if r3 else None

    def available(self) -> bool:
        return find_rubberband() is not None

    def _require(self) -> str:
        exe = find_rubberband()
        if exe is None:
            raise BackendUnavailable(
                "the 'rubberband' program was not found.\n"
                "It is required for non-uniform compression -- the core feature.\n"
                "  Windows: download the command-line utility from\n"
                "           https://breakfastquay.com/rubberband/\n"
                "           and drop rubberband.exe into this project's bin/ folder\n"
                "  macOS:   brew install rubberband\n"
                "  Linux:   apt install rubberband-cli"
            )
        import pyrubberband.pyrb as _pyrb
        # setattr, not attribute assignment: this is inside a class body, so a
        # literal `_pyrb.__RUBBERBAND_UTIL = exe` would name-mangle to
        # `_RubberBandStretcher__RUBBERBAND_UTIL` and silently do nothing.
        if getattr(_pyrb, "__RUBBERBAND_UTIL", None) != exe:
            setattr(_pyrb, "__RUBBERBAND_UTIL", exe)
        return exe

    def stretch(self, y, sr, rate):
        self._require()
        import pyrubberband as pyrb
        return pyrb.time_stretch(_safe(y), sr, rate, rbargs=self.rbargs).astype(np.float32)

    def stretch_map(self, y, sr, time_map):
        self._require()
        import pyrubberband as pyrb
        pairs = time_map.as_pairs() if hasattr(time_map, "as_pairs") else list(time_map)
        if pairs[-1][0] != len(y):
            raise ValueError(
                f"time map must end at the input length: got {pairs[-1][0]}, expected {len(y)}")
        return pyrb.timemap_stretch(_safe(y), sr, pairs, rbargs=self.rbargs).astype(np.float32)


BACKENDS: dict[str, type[TimeStretcher]] = {
    "resample": ResampleStretcher,
    "phasevocoder": PhaseVocoderStretcher,
    "wsola": WSOLAStretcher,
    "rubberband": RubberBandStretcher,
}


def get_backend(name: str, **kw) -> TimeStretcher:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kw)
