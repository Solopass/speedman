"""Forced-choice listening test: the instrument of last resort.

ASR saturates above ~3.5x and the modulation metric structurally favours the uniform
control, so neither can settle whether Speedman's non-uniform warping helps at the 5-6x
speeds it actually ships at. Human ears can.

Each trial plays the same clip twice -- Speedman and a uniform stretch to the same
duration, order randomised -- and collects two judgements in one listening pass:

    choice     which was easier to understand   -> does Speedman help at this speed
    followed   could you follow the content     -> where speech stops being usable

Running the speeds as a randomised ladder of these trials answers both questions at once.
Order is randomised across rungs rather than ascending because an ascending ladder
confounds speed with practice: by the last rung the listener has heard the content several
times already.

BLINDING. The slot -> condition mapping lives here and in manifest.json on disk, and is
withheld from every payload until the session is complete. Rendered files are named by
trial and slot, never by condition, so the blind survives someone opening devtools or
listing the directory.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from speedman import io as sio
from speedman.config import build_config
from speedman.pipeline import process

from . import manifest

LISTENING_DIR = Path(__file__).resolve().parent / "listening"
SR = 24000

SPEEDMAN = "speedman"
UNIFORM = "uniform"
CONDITIONS = (SPEEDMAN, UNIFORM)

CHOICES = ("a", "b", "none")
"""'none' is a real answer, not a cop-out: forcing a pick when there is no audible
difference manufactures signal out of coin flips."""

FOLLOWED = ("both", "picked", "neither")
"""both = followed the content either way; picked = only the one chosen; neither = lost."""

DEFAULT_EXCERPT_S = 30.0
DEFAULT_SPEEDS = (3.0, 4.0, 5.0, 6.0, 8.0)
DEFAULT_TRIALS_PER_SPEED = 6


class SessionNotFound(RuntimeError):
    pass


class InvalidVerdict(ValueError):
    pass


@dataclass
class Trial:
    index: int
    speed: float
    clip_label: str
    source: str
    offset_s: float
    duration_s: float
    slot_a: str  # which condition is in slot A -- never leaves the server until reveal
    slot_b: str

    def condition_for(self, slot: str) -> str:
        if slot == "a":
            return self.slot_a
        if slot == "b":
            return self.slot_b
        raise InvalidVerdict(f"unknown slot {slot!r}; expected 'a' or 'b'")

    def slot_for(self, condition: str) -> str:
        return "a" if self.slot_a == condition else "b"

    def blind(self) -> dict[str, Any]:
        """What the client is allowed to see. Deliberately omits slot_a/slot_b."""
        return {
            "index": self.index,
            "speed": self.speed,
            "duration_s": round(self.duration_s / self.speed, 2),
        }


@dataclass
class Verdict:
    trial_index: int
    choice: str       # "a" | "b" | "none"
    followed: str     # "both" | "picked" | "neither"
    recorded_at: float = field(default_factory=time.time)

    def winner(self, trial: Trial) -> Optional[str]:
        """Which condition the listener preferred, or None for 'no difference'."""
        if self.choice == "none":
            return None
        return trial.condition_for(self.choice)


@dataclass
class Session:
    session_id: str
    created_at: float
    trials: list[Trial]
    verdicts: dict[int, Verdict] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return LISTENING_DIR / self.session_id

    def is_complete(self) -> bool:
        return len(self.verdicts) >= len(self.trials)

    def next_index(self) -> Optional[int]:
        for t in self.trials:
            if t.index not in self.verdicts:
                return t.index
        return None

    def audio_path(self, trial_index: int, slot: str) -> Path:
        if slot not in ("a", "b"):
            raise InvalidVerdict(f"unknown slot {slot!r}; expected 'a' or 'b'")
        return self.directory / f"trial_{trial_index:03d}_{slot}.wav"

    def blind_state(self) -> dict[str, Any]:
        """Everything the client needs to run the session, and nothing that unblinds it."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_trials": len(self.trials),
            "completed": len(self.verdicts),
            "next_index": self.next_index(),
            "complete": self.is_complete(),
            "trials": [t.blind() for t in self.trials],
            "verdicts": [
                {"trial_index": v.trial_index, "choice": v.choice, "followed": v.followed}
                for v in sorted(self.verdicts.values(), key=lambda v: v.trial_index)
            ],
        }


# --------------------------------------------------------------------------- building

def _excerpt(clip: manifest.Clip, excerpt_s: float, rng: random.Random) -> tuple[np.ndarray, float]:
    """A short window from inside a manifest clip.

    Manifest clips are 90s, which at 4x is a 22s trial -- far too long for a test that
    needs 30 of them. Returns (audio, offset_within_source).
    """
    y = clip.load(SR)
    want = int(excerpt_s * SR)
    if len(y) <= want:
        return y, clip.offset_s
    start = rng.randint(0, len(y) - want)
    return y[start:start + want], clip.offset_s + start / SR


def build_session(
    speeds: Iterable[float] = DEFAULT_SPEEDS,
    trials_per_speed: int = DEFAULT_TRIALS_PER_SPEED,
    clips: Optional[list[manifest.Clip]] = None,
    excerpt_s: float = DEFAULT_EXCERPT_S,
    seed: Optional[int] = None,
) -> Session:
    """Render every trial up front and persist the session.

    Rendering ahead of time keeps the test responsive -- a pause between trials while the
    machine works would leak timing information about which condition is which.
    """
    speeds = [float(s) for s in speeds]
    if not speeds:
        raise ValueError("need at least one speed")
    if trials_per_speed < 1:
        raise ValueError("need at least one trial per speed")

    available, _ = manifest.available(clips if clips is not None else manifest.CLIPS)
    if not available:
        raise ValueError("no clips available -- check eval/manifest.py source paths")

    rng = random.Random(seed)
    session_id = f"lt_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session = Session(session_id=session_id, created_at=time.time(), trials=[])
    session.directory.mkdir(parents=True, exist_ok=True)

    specs = []
    for speed in speeds:
        for i in range(trials_per_speed):
            specs.append((speed, available[(i + int(speed)) % len(available)]))

    # Randomise across rungs so practice and fatigue do not load onto the fastest speeds.
    rng.shuffle(specs)

    for index, (speed, clip) in enumerate(specs):
        audio, offset = _excerpt(clip, excerpt_s, rng)
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)
        trial = Trial(
            index=index,
            speed=speed,
            clip_label=clip.label,
            source=str(clip.source),
            offset_s=offset,
            duration_s=len(audio) / SR,
            slot_a=conditions[0],
            slot_b=conditions[1],
        )
        for slot in ("a", "b"):
            rendered = _render(audio, speed, trial.condition_for(slot))
            sio.save(session.audio_path(index, slot), rendered, SR)
        session.trials.append(trial)

    _save_manifest(session)
    _save_verdicts(session)
    return session


def _render(y: np.ndarray, speed: float, condition: str) -> np.ndarray:
    from dataclasses import replace

    cfg = build_config(speed=speed, preset="fast", backend="rubberband")
    if condition == UNIFORM:
        cfg = replace(cfg, uniform=True)
    return process(y, SR, cfg).audio


# --------------------------------------------------------------------------- persistence

def _save_manifest(session: Session) -> None:
    payload = {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "trials": [asdict(t) for t in session.trials],
    }
    (session.directory / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _save_verdicts(session: Session) -> None:
    payload = [asdict(v) for v in sorted(session.verdicts.values(), key=lambda v: v.trial_index)]
    (session.directory / "verdicts.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def load_session(session_id: str) -> Session:
    """Load a session by id. Resuming after a browser crash must not lose verdicts."""
    safe = Path(str(session_id)).name  # never let an id escape LISTENING_DIR
    directory = LISTENING_DIR / safe
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SessionNotFound(f"no listening session {session_id!r}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = Session(
        session_id=data["session_id"],
        created_at=data["created_at"],
        trials=[Trial(**t) for t in data["trials"]],
    )

    verdicts_path = directory / "verdicts.json"
    if verdicts_path.is_file():
        for v in json.loads(verdicts_path.read_text(encoding="utf-8")):
            session.verdicts[v["trial_index"]] = Verdict(**v)
    return session


def record_verdict(session: Session, trial_index: int, choice: str, followed: str) -> Verdict:
    choice = str(choice).strip().lower()
    followed = str(followed).strip().lower()
    if choice not in CHOICES:
        raise InvalidVerdict(f"choice must be one of {list(CHOICES)}, got {choice!r}")
    if followed not in FOLLOWED:
        raise InvalidVerdict(f"followed must be one of {list(FOLLOWED)}, got {followed!r}")
    if not any(t.index == trial_index for t in session.trials):
        raise InvalidVerdict(f"session has no trial {trial_index}")
    if choice == "none" and followed == "picked":
        raise InvalidVerdict("'picked' is meaningless when the choice is 'no difference'")

    verdict = Verdict(trial_index=trial_index, choice=choice, followed=followed)
    session.verdicts[trial_index] = verdict
    _save_verdicts(session)
    return verdict


def list_sessions() -> list[dict[str, Any]]:
    if not LISTENING_DIR.is_dir():
        return []
    out = []
    for d in sorted(LISTENING_DIR.iterdir(), reverse=True):
        if not (d / "manifest.json").is_file():
            continue
        try:
            s = load_session(d.name)
        except Exception:
            continue
        out.append({
            "session_id": s.session_id,
            "created_at": s.created_at,
            "total_trials": len(s.trials),
            "completed": len(s.verdicts),
            "complete": s.is_complete(),
            "speeds": sorted({t.speed for t in s.trials}),
        })
    return out


# --------------------------------------------------------------------------- analysis

def reveal(session: Session) -> dict[str, Any]:
    """Unblind and summarise. Refuses while trials remain, so a mid-session peek cannot
    bias the trials that are left."""
    if not session.is_complete():
        raise InvalidVerdict(
            f"session is {len(session.verdicts)}/{len(session.trials)} complete; "
            "finish it before revealing")
    return {
        "session_id": session.session_id,
        "per_speed": summarise(session),
        "trials": [
            {
                **t.blind(),
                "clip_label": t.clip_label,
                "slot_a": t.slot_a,
                "slot_b": t.slot_b,
                "choice": session.verdicts[t.index].choice,
                "winner": session.verdicts[t.index].winner(t),
                "followed": session.verdicts[t.index].followed,
            }
            for t in session.trials if t.index in session.verdicts
        ],
    }


def summarise(session: Session) -> list[dict[str, Any]]:
    """Per speed: Speedman wins, uniform wins, ties, and how often content was followable."""
    by_speed: dict[float, dict[str, int]] = {}
    for trial in session.trials:
        verdict = session.verdicts.get(trial.index)
        if verdict is None:
            continue
        bucket = by_speed.setdefault(trial.speed, {
            "speedman": 0, "uniform": 0, "ties": 0,
            "followed_both": 0, "followed_picked": 0, "followed_neither": 0,
        })
        winner = verdict.winner(trial)
        if winner is None:
            bucket["ties"] += 1
        else:
            bucket[winner] += 1
        bucket[f"followed_{verdict.followed}"] += 1

    return [
        {
            "speed": speed,
            **counts,
            "n_decisive": counts["speedman"] + counts["uniform"],
            "n_trials": counts["speedman"] + counts["uniform"] + counts["ties"],
        }
        for speed, counts in sorted(by_speed.items())
    ]
