"""Annotation sources: turn audio into typed, timed spans.

ALL ANALYSIS RUNS ON THE ORIGINAL AUDIO. VAD and onset detection are reliable
at 1x and unreliable on compressed audio -- running them post-compression was
one of the original draft's core mistakes.

The `Annotation` structure is deliberately source-agnostic: the `vad` source
below is the only implementation today, but a forced-alignment source
(docs/TRANSCRIPT_ALIGNMENT.md) must be droppable in without touching ratemap.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SpanKind(str, Enum):
    SILENCE = "silence"   # ratemap decides boundary-vs-short by duration
    ONSET = "onset"       # protected: consonant transients / syllable onsets
    STEADY = "steady"     # crushed: redundant vowel steady-state
    SPEECH = "speech"     # everything else


@dataclass(frozen=True)
class Span:
    start: int   # inclusive sample index, INPUT domain
    end: int     # exclusive
    kind: SpanKind

    @property
    def length(self) -> int:
        return self.end - self.start

    def duration(self, sr: int) -> float:
        return self.length / sr


@dataclass
class Annotation:
    spans: list[Span]
    sr: int
    n_samples: int
    meta: dict = field(default_factory=dict)

    def silence_fraction(self) -> float:
        """Predicts how much lever 1 will deliver on this file, so it is worth
        reporting to the user: speech ends up at roughly N*(1-0.6*s)."""
        sil = sum(s.length for s in self.spans if s.kind is SpanKind.SILENCE)
        return sil / max(1, self.n_samples)

    def validate(self) -> None:
        if not self.spans:
            raise ValueError("annotation has no spans")
        if self.spans[0].start != 0 or self.spans[-1].end != self.n_samples:
            raise ValueError("spans must tile the whole signal")
        for a, b in zip(self.spans, self.spans[1:]):
            if a.end != b.start:
                raise ValueError(f"gap or overlap between spans at sample {a.end}")


# ----------------------------------------------------------------------------- VAD
#
# Everything below works on FRAMES, never samples. A 10-hour audiobook is ~860M
# samples; a per-sample array (let alone a Python loop over one) is gigabytes and
# minutes. At hop=256 the same file is 3.4M frames, which is ordinary numpy work.

HOP = 256  # ~10.7 ms at 24 kHz; shared with librosa's onset hop so both land on
           # the same grid and no resampling between them is needed.


def _frame_db(y: np.ndarray, sr: int, hop: int = HOP, frame_ms: float = 20.0) -> np.ndarray:
    """Frame-wise RMS in dB. Vectorised; no Python loop over samples."""
    frame = max(hop, int(sr * frame_ms / 1000))
    if len(y) < frame:
        return np.array([-120.0])
    # Prefix sums: O(n) memory and one pass. Fancy-indexing a (n_frames, frame)
    # matrix here cost hundreds of MB on a 10-minute file.
    cs = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    starts = np.arange(0, len(y) - frame + 1, hop)
    ms = (cs[starts + frame] - cs[starts]) / frame
    return 20 * np.log10(np.sqrt(np.maximum(ms, 0.0)) + 1e-12)


def _hysteresis(db: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Schmitt trigger over frames.

    Vectorised: a frame is speech if the most recent threshold crossing was an
    upward one, which is a forward-fill of the last decisive frame.
    """
    up, down = db > hi, db < lo
    state = np.where(up, 1, np.where(down, 0, -1))
    idx = np.where(state >= 0, np.arange(len(state)), 0)
    np.maximum.accumulate(idx, out=idx)
    return state[idx] == 1


def _min_run(mask: np.ndarray, min_frames: int, fill: bool) -> np.ndarray:
    """Fill runs of `not fill` shorter than min_frames. Vectorised over runs."""
    if min_frames <= 1 or mask.size == 0:
        return mask
    m = mask.copy()
    change = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [len(m)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if m[a] != fill and (b - a) < min_frames and a > 0 and b < len(m):
            m[a:b] = fill
    return m


def detect_speech_frames(y: np.ndarray, sr: int, hop: int = HOP,
                         offset_db: float = 8.0, min_silence_ms: float = 60.0,
                         min_speech_ms: float = 50.0) -> np.ndarray:
    """Boolean speech mask, one entry per frame.

    Energy VAD with an adaptive floor and hysteresis. Deliberately not silero:
    that pulls in torch (~2 GB) for a job whose only real requirement is finding
    pauses longer than ~250 ms, which energy does reliably. Swapping in a neural
    source later is a contained change -- that is what Annotation is for.
    """
    db = _frame_db(y, sr, hop)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    thresh = max(floor + offset_db, peak - 35.0)
    mask = _hysteresis(db, thresh, thresh - 3.0)
    frame_ms = hop / sr * 1000
    mask = _min_run(mask, int(min_silence_ms / frame_ms), fill=True)
    mask = _min_run(mask, int(min_speech_ms / frame_ms), fill=False)
    return mask


def detect_speech(y: np.ndarray, sr: int, **kw) -> np.ndarray:
    """Per-sample convenience wrapper. Fine for short signals and tests; the
    pipeline itself never expands to samples."""
    mask = detect_speech_frames(y, sr, **kw)
    return np.repeat(mask, HOP)[: len(y)] if mask.size else np.zeros(len(y), bool)


def onset_strength_frames(y: np.ndarray, sr: int, hop: int = HOP,
                          block_sec: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """Onset-strength envelope and detected onset positions, both in frames.

    Computed in overlapping blocks. librosa's STFT allocates a complex spectrum
    for the whole signal at once -- roughly 0.5-1 GB per 10 minutes of audio --
    which is the single largest allocation in the pipeline and the thing most
    likely to kill a long file. Blocking bounds it at a constant.
    """
    import librosa

    block = int(block_sec * sr)
    if len(y) <= block:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    else:
        pad = 4 * hop          # discarded margin; onset strength is differential
        chunks = []
        for start in range(0, len(y), block):
            a = max(0, start - pad)
            seg = y[a:start + block + pad]
            e = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop)
            lead = (start - a) // hop
            want = min(block // hop, len(y[start:start + block]) // hop + 1)
            chunks.append(e[lead:lead + want])
        env = np.concatenate(chunks)

    n_frames = 1 + len(y) // hop
    if len(env) < n_frames:
        env = np.pad(env, (0, n_frames - len(env)), mode="edge")
    env = env[:n_frames]

    delta = 0.15 * float(np.mean(env) + 1e-9)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=delta, wait=3)
    return env, np.asarray(peaks, dtype=int)


def annotate_vad(y: np.ndarray, sr: int, protect_window_ms: float = 120.0,
                 protect_lead_ms: float = 30.0, steady_percentile: float = 35.0,
                 hop: int = HOP) -> Annotation:
    """The default annotation source.

    Protection windows are SYLLABLE-WIDTH, not phoneme-width: the backend places
    time-map windows with ~5 ms jitter, so a window narrower than ~80 ms lands on
    the wrong audio (docs/SPIKE_RESULTS.md). Do not narrow them without re-running
    eval/spikes/registration_spike.py.
    """
    n = len(y)
    speech = detect_speech_frames(y, sr, hop)
    env, onsets = onset_strength_frames(y, sr, hop)

    n_frames = min(len(speech), len(env))
    speech, env = speech[:n_frames], env[:n_frames]
    if n_frames == 0:
        return Annotation([Span(0, n, SpanKind.SPEECH)], sr, n, {"source": "vad"})

    SIL, SPE, STE, ONS = 0, 1, 2, 3
    kinds = np.where(speech, SPE, SIL).astype(np.int8)

    # Steady state: low spectral flux inside speech.
    if speech.any():
        thresh = np.percentile(env[speech], steady_percentile)
        kinds[speech & (env < thresh)] = STE

    # Protection windows last, so they win over STEADY.
    lead = max(1, int(sr * protect_lead_ms / 1000 / hop))
    width = max(1, int(sr * protect_window_ms / 1000 / hop))
    if onsets.size:
        starts = np.clip(onsets - lead, 0, n_frames)
        ends = np.clip(starts + width, 0, n_frames)
        # paint via a difference array: O(onsets), not O(onsets * width)
        marks = np.zeros(n_frames + 1, np.int32)
        np.add.at(marks, starts, 1)
        np.add.at(marks, ends, -1)
        covered = np.cumsum(marks[:-1]) > 0
        kinds[covered & speech] = ONS

    # Frame runs -> sample-domain spans, vectorised.
    change = np.flatnonzero(np.diff(kinds)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n_frames]])
    lut = {SIL: SpanKind.SILENCE, SPE: SpanKind.SPEECH,
           STE: SpanKind.STEADY, ONS: SpanKind.ONSET}
    spans = [Span(int(a) * hop, int(b) * hop, lut[int(kinds[a])])
             for a, b in zip(starts, ends)]
    spans[-1] = Span(spans[-1].start, n, spans[-1].kind)   # cover the tail
    spans = [s for s in spans if s.length > 0]

    ann = Annotation(spans=spans, sr=sr, n_samples=n,
                     meta={"n_onsets": int(onsets.size), "source": "vad"})
    ann.validate()
    return ann
