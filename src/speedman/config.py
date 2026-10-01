"""Configuration dataclasses.

All tunable numbers live here. Defaults are starting points chosen from the
reasoning in docs/ARCHITECTURE.md, NOT tuned values -- they are placeholders
until eval/results.md says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RateMapConfig:
    """Shape of the non-uniform time map.

    Multipliers are *relative*; the map is normalised so the whole file hits the
    requested average speed exactly, so raising one multiplier makes everything
    else slower, not the file shorter.
    """

    # --- lever 1: pause compression ---
    silence_mult: float = 2.5
    """How much harder boundary pauses compress than speech."""

    boundary_ms: float = 250.0
    """Silences longer than this are prosodic boundaries: floored and compressed
    hard. SHORTER GAPS ARE NOT FLOORED -- flooring every gap makes lever 1
    net-harmful in the 4-8x band (ARCHITECTURE.md, the inversion table)."""

    floor_ms: float = 25.0
    """Minimum surviving duration of a boundary pause, in OUTPUT time."""

    # --- lever 2: within-speech reallocation ---
    protect_mult: float = 0.7
    """q -- transient regions run this much slower than the base rate."""

    crush_mult: float = 2.0
    """Vowel steady-state regions run this much faster than the base rate."""

    protect_window_ms: float = 120.0
    """Width of a protection window. MUST stay >= ~80 ms: the backend places
    time-map windows with ~5 ms jitter, so a narrower window lands on the wrong
    audio (docs/SPIKE_RESULTS.md). This is syllable-onset granularity, not
    phoneme granularity, and that is a measured constraint rather than a
    preference."""

    protect_lead_ms: float = 30.0
    """How far before the detected onset the window opens."""

    # --- safety rails ---
    max_rate: float = 40.0
    """Hard ceiling on instantaneous rate; keeps the backend inside a regime it
    tolerates. Regions hitting this are clamped and the remainder re-solved."""

    min_rate: float = 0.25
    anchor_ms: float = 10.0
    """Time-map anchor spacing. 10 ms is the measured optimum -- finer buys
    nothing because the residual is the backend's own windowing jitter."""


@dataclass(frozen=True)
class PostConfig:
    presence_db: float = 3.0
    """Additive presence EQ gain. Bounded at +4 dB by ARCHITECTURE.md."""
    presence_lo: float = 2000.0
    presence_hi: float = 5000.0

    warmth_db: float = 0.0
    """Low-mid vocal warmth/body lift at 280 Hz to counterbalance presence and prevent thin tone."""
    warmth_freq: float = 280.0
    warmth_q: float = 1.0

    highpass_hz: float = 0.0
    """Sub-bass high-pass filter cutoff to eliminate mic rumble and plosives. 0 to disable."""

    deess_db: float = 0.0
    """Dynamic de-esser attenuation ceiling (dB) targeting sibilance (5.5-8.5 kHz). 0 to disable."""
    deess_lo: float = 5500.0
    deess_hi: float = 8500.0

    expander_db: float = 0.0
    """Downward expansion in pauses to quiet room hiss. 0 to disable."""

    transient_db: float = 0.0
    """Onset-gated transient enhancement. Off by default -- it ships only if the
    eval table says it helps."""

    drc_threshold_db: float = -24.0
    drc_ratio: float = 2.0
    """Gentle downward compression so unstressed syllables survive between loud
    stressed ones at speed."""

    target_lufs: float = -16.0
    limiter_ceiling: float = 0.97
    """A limiter is REQUIRED: EQ + transient boost + normalisation together will
    otherwise exceed full scale (ARCHITECTURE.md)."""


@dataclass(frozen=True)
class Config:
    speed: float = 5.0
    sample_rate: int = 24000
    backend: str = "rubberband"
    engine: str = "r2"
    crispness: int | None = None
    staged: bool = False
    uniform: bool = False
    """Disable non-uniform compression -- the control condition."""
    ratemap: RateMapConfig = RateMapConfig()
    post: PostConfig = PostConfig()


PRESETS: dict[str, dict] = {
    # Conservative: barely-there reallocation, gentle post chain with warmth & rumble cut.
    "natural": dict(
        ratemap=RateMapConfig(silence_mult=2.0, protect_mult=0.85, crush_mult=1.4),
        post=PostConfig(presence_db=1.5, warmth_db=1.0, highpass_hz=60.0, deess_db=1.5, drc_ratio=1.5),
    ),
    # The daily driver: clarity with vocal body warmth, sibilance taming & rumble highpass.
    "fast": dict(
        ratemap=RateMapConfig(),
        post=PostConfig(presence_db=3.0, warmth_db=1.5, highpass_hz=70.0, deess_db=2.5),
    ),
    # High-fidelity studio mode: R3 fine engine, de-esser, rumble cut & vocal warmth.
    "audiophile": dict(
        backend="rubberband",
        engine="r3",
        crispness=5,
        ratemap=RateMapConfig(silence_mult=2.2, protect_mult=0.8, crush_mult=1.6),
        post=PostConfig(
            presence_db=2.5,
            warmth_db=2.0,
            highpass_hz=80.0,
            deess_db=4.0,
            drc_ratio=1.8,
            drc_threshold_db=-24.0,
        ),
    ),
    "aggressive": dict(
        ratemap=RateMapConfig(silence_mult=3.0, protect_mult=0.6, crush_mult=2.5),
        post=PostConfig(presence_db=4.0, warmth_db=1.5, highpass_hz=80.0, deess_db=3.5, drc_ratio=3.0),
    ),
    "max": dict(
        ratemap=RateMapConfig(silence_mult=3.5, protect_mult=0.5, crush_mult=3.0,
                              floor_ms=20.0),
        post=PostConfig(presence_db=4.0, warmth_db=2.0, highpass_hz=90.0, deess_db=4.5,
                        drc_ratio=3.5, drc_threshold_db=-28.0),
    ),
}



def build_config(speed: float, preset: str = "fast", **overrides) -> Config:
    """Preset supplies the parameter shape; --speed always wins on speed itself."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = Config(speed=speed, **PRESETS[preset])
    return replace(cfg, **overrides) if overrides else cfg
