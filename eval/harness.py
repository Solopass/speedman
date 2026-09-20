"""Run the condition grid and write eval/results.md -- the file config.py defers to.

    python -m eval.harness                       # default grid
    python -m eval.harness --speeds 5 --presets fast
    python -m eval.harness --clips podcast_dating --no-cache

What it measures: for every (clip, speed, condition), the word error rate of ASR on
the processed audio against ASR on the SAME clip unprocessed. Using a 1x ASR pass as
the reference rather than a human transcript isolates the damage compression does from
the engine's own baseline error -- a word parakeet gets wrong at 1x is not a cost
Speedman imposed.

The `uniform` condition is the control that matters: it is a constant-rate stretch to
the same duration. Speedman's entire claim is that it beats that at the same speed, so
any preset that does not is worth knowing about.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from speedman import io as sio
from speedman.config import PRESETS, build_config
from speedman.pipeline import process

from . import asr, manifest
from .wer import wer, cer

EVAL_DIR = Path(__file__).resolve().parent
WORK_DIR = EVAL_DIR / "clips"
CACHE_PATH = EVAL_DIR / "cache.json"
RESULTS_MD = EVAL_DIR / "results.md"
RESULTS_JSON = EVAL_DIR / "results.json"

SR = 24000
UNIFORM = "uniform"

# Above this WER the scoring engine has essentially stopped transcribing, and
# differences between conditions are noise rather than signal. Measured on parakeet:
# WER climbs 0.06 -> 0.40 -> 0.82 across 2.5x/3x/3.5x, so the instrument's useful
# range ends around 3.5x. Results past it are reported but flagged, never ranked.
SATURATION_WER = 0.85

# The band where the metric discriminates. Speeds outside it still run on request;
# they just cannot settle a tuning question.
DEFAULT_SPEEDS = "2,2.5,3,3.5"


@dataclass
class Row:
    clip: str
    kind: str
    speed: float
    condition: str
    wer: float
    cer: float
    ref_words: int
    hyp_words: int
    in_duration_s: float
    out_duration_s: float
    silence_fraction: float
    reported_speech_rate: float
    measured_speech_rate: float
    process_s: float


def _cache_load(use_cache: bool) -> dict:
    if use_cache and CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _cache_save(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _measured_speech_rate(y: np.ndarray, cfg, speed: float) -> float:
    """What the time map actually runs non-silence at.

    Now the same code production uses -- `pipeline.process` reports this as
    `effective_speech_rate`, so the harness's reported-vs-measured comparison should read
    ~0 and stay there. It is kept as a separate path so a regression in the pipeline shows
    up as a gap rather than being invisible.
    """
    from speedman import analyze, ratemap

    try:
        ann = analyze.annotate_vad(
            y, SR,
            protect_window_ms=cfg.ratemap.protect_window_ms,
            protect_lead_ms=cfg.ratemap.protect_lead_ms,
        )
        tm = ratemap.build_time_map(ann, speed, cfg.ratemap)
        return tm.measured_speech_rate([sp.kind.value for sp in ann.spans])
    except Exception:
        return float("nan")


def _render(y: np.ndarray, speed: float, condition: str, out_path: Path) -> tuple[dict, float]:
    """Process one condition and write the audio. Returns (notes, elapsed)."""
    uniform = condition == UNIFORM
    preset = "fast" if uniform else condition
    cfg = build_config(speed=speed, preset=preset, backend="rubberband", uniform=uniform)

    t0 = time.perf_counter()
    res = process(y, SR, cfg)
    elapsed = time.perf_counter() - t0

    sio.save(out_path, res.audio, res.sr)
    notes = dict(res.notes)
    notes["out_duration_s"] = len(res.audio) / res.sr
    notes["measured_speech_rate"] = float("nan") if uniform else _measured_speech_rate(y, cfg, speed)
    return notes, elapsed


def run(clips, speeds, conditions, engine: str, use_cache: bool) -> list[Row]:
    asr.require_online()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_load(use_cache)
    rows: list[Row] = []

    total = len(clips) * len(speeds) * len(conditions)
    done = 0

    for clip in clips:
        fp = clip.fingerprint()
        print(f"\n[{clip.label}] {clip.kind}, {clip.duration_s:.0f}s from "
              f"{clip.source.name[:46]} @ {clip.offset_s:.0f}s", flush=True)

        # Decoding is the slowest step for a multi-hour source, so do it lazily: a fully
        # cached clip never touches the audio at all.
        _audio: list[np.ndarray] = []

        def get_audio() -> np.ndarray:
            if not _audio:
                _audio.append(clip.load(SR))
            return _audio[0]

        # Reference: ASR on the untouched clip.
        ref_key = f"{fp}|ref|{engine}"
        if ref_key in cache:
            ref_text = cache[ref_key]
            print(f"  reference: cached ({len(ref_text.split())} words)")
        else:
            try:
                ref_path = WORK_DIR / f"{clip.label}_{fp}_1x.wav"
                sio.save(ref_path, get_audio(), SR)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"  SKIP: {e}")
                continue
            ref_text = asr.transcribe(ref_path, engine=engine).text
            cache[ref_key] = ref_text
            _cache_save(cache)
            print(f"  reference: {len(ref_text.split())} words transcribed at 1x")

        if not ref_text.strip():
            print("  SKIP: reference transcript is empty (no speech detected?)")
            continue

        for speed in speeds:
            for condition in conditions:
                done += 1
                key = f"{fp}|{speed}|{condition}|{engine}"
                tag = f"{speed:g}x {condition}"

                if key in cache:
                    entry = cache[key]
                    print(f"  [{done:>3}/{total}] {tag:<20} cached   WER {entry['wer']:.3f}")
                else:
                    out_path = WORK_DIR / f"{clip.label}_{fp}_{speed:g}x_{condition}.wav"
                    notes, elapsed = _render(get_audio(), speed, condition, out_path)
                    hyp_text = asr.transcribe(out_path, engine=engine).text
                    w = wer(ref_text, hyp_text)
                    c = cer(ref_text, hyp_text)
                    entry = {
                        "wer": w.rate,
                        "cer": c.rate,
                        "ref_words": w.length,
                        "hyp_words": len(hyp_text.split()),
                        "out_duration_s": notes["out_duration_s"],
                        "silence_fraction": float(notes.get("silence_fraction", 0.0)),
                        "reported_speech_rate": float(notes.get("effective_speech_rate", speed)),
                        "measured_speech_rate": float(notes.get("measured_speech_rate", float("nan"))),
                        "process_s": elapsed,
                    }
                    cache[key] = entry
                    _cache_save(cache)
                    print(f"  [{done:>3}/{total}] {tag:<20} WER {w.rate:.3f}  "
                          f"CER {c.rate:.3f}  ({elapsed:.1f}s)")

                rows.append(Row(
                    clip=clip.label, kind=clip.kind, speed=speed, condition=condition,
                    wer=entry["wer"], cer=entry["cer"],
                    ref_words=entry["ref_words"], hyp_words=entry["hyp_words"],
                    in_duration_s=clip.duration_s,
                    out_duration_s=entry["out_duration_s"],
                    silence_fraction=entry["silence_fraction"],
                    reported_speech_rate=entry["reported_speech_rate"],
                    measured_speech_rate=entry["measured_speech_rate"],
                    process_s=entry["process_s"],
                ))

    return rows


def _mean(values) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return statistics.fmean(vals) if vals else float("nan")


def summarise(rows: list[Row], speeds, conditions) -> str:
    """Build results.md. The headline table is preset-vs-uniform at each speed,
    because that is the comparison the product claim rests on."""
    if not rows:
        return "# Speedman Evaluation\n\nNo results: no clips ran.\n"

    out: list[str] = []
    out.append("# Speedman Evaluation Results")
    out.append("")
    out.append(f"Generated by `python -m eval.harness` on {time.strftime('%Y-%m-%d %H:%M')}.")
    out.append("")
    out.append("**Metric.** Word error rate of ASR on processed audio, against ASR on the same "
               "clip at 1x. A 1x reference isolates compression damage from the engine's own "
               "baseline error. Lower is better; 0.000 would mean compression cost nothing.")
    out.append("")
    out.append("**Caveat.** The scoring engine (parakeet) was not trained on time-compressed "
               "speech, so absolute WER overstates how unintelligible a clip is to a human. "
               "Read the *differences between conditions*, not the absolute numbers.")
    out.append("")

    clips = sorted({r.clip for r in rows})
    out.append(f"Clips: {len(clips)} ({', '.join(clips)}) | "
               f"Speeds: {', '.join(f'{s:g}x' for s in speeds)} | "
               f"Conditions: {', '.join(conditions)}")
    out.append("")

    # ---- headline: does any preset beat the uniform control? ----
    out.append("## Headline: presets vs the uniform control")
    out.append("")
    out.append("`uniform` is a constant-rate stretch to the same duration -- naive speed-up. "
               "A preset earns its complexity only by scoring below it.")
    out.append("")
    non_uniform = [c for c in conditions if c != UNIFORM]
    header = "| speed | " + " | ".join(f"`{c}`" for c in non_uniform) + " | `uniform` | best |"
    out.append(header)
    out.append("|" + "---|" * (len(non_uniform) + 3))

    saturated_speeds = []
    for speed in speeds:
        cells, scores = [], {}
        for cond in non_uniform:
            v = _mean(r.wer for r in rows if r.speed == speed and r.condition == cond)
            scores[cond] = v
            cells.append(f"{v:.3f}" if v == v else "--")
        uni = _mean(r.wer for r in rows if r.speed == speed and r.condition == UNIFORM)
        scores[UNIFORM] = uni
        valid = {k: v for k, v in scores.items() if v == v}
        uni_cell = f"{uni:.3f}" if uni == uni else "--"

        # Once every condition is at the ceiling, naming a winner would be reading noise.
        if valid and min(valid.values()) >= SATURATION_WER:
            saturated_speeds.append(speed)
            out.append(f"| {speed:g}x | " + " | ".join(cells) +
                       f" | {uni_cell} | _metric saturated_ |")
            continue

        best = min(valid, key=valid.get) if valid else "--"
        delta = ""
        if best != UNIFORM and best in valid and uni == uni and uni > 0:
            delta = f" ({(valid[best] - uni) / uni * 100:+.0f}% vs uniform)"
        out.append(f"| {speed:g}x | " + " | ".join(cells) +
                   f" | {uni_cell} | **{best}**{delta} |")
    out.append("")

    if saturated_speeds:
        out.append(f"> At {', '.join(f'{s:g}x' for s in saturated_speeds)} every condition scores "
                   f"at or above WER {SATURATION_WER}: the scoring engine has stopped transcribing, "
                   "so those rows cannot rank anything. They are shown for completeness only. "
                   "Validating the 5-6x target band needs either a listening test or an ASR model "
                   "robust to time-compressed speech.")
        out.append("")

    # ---- per-clip detail ----
    out.append("## Per-clip detail")
    out.append("")
    out.append("| clip | kind | speed | condition | WER | CER | out dur | reported rate | measured rate |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r.clip, r.speed, r.condition)):
        meas = f"{r.measured_speech_rate:.2f}x" if r.measured_speech_rate == r.measured_speech_rate else "--"
        out.append(f"| {r.clip} | {r.kind} | {r.speed:g}x | {r.condition} | {r.wer:.3f} | "
                   f"{r.cer:.3f} | {r.out_duration_s:.1f}s | {r.reported_speech_rate:.2f}x | {meas} |")
    out.append("")

    # ---- the reported-vs-measured gap ----
    gaps = [(r.reported_speech_rate - r.measured_speech_rate) / r.measured_speech_rate * 100
            for r in rows if r.measured_speech_rate == r.measured_speech_rate
            and r.measured_speech_rate > 0]
    if gaps:
        out.append("## Reported vs measured speech rate")
        out.append("")
        out.append("`effective_speech_rate` is now read off the solved time map, so this gap "
                   "should sit at ~0 and is a regression check rather than a finding. It was "
                   "-0.96% mean and -4.21% worst when the pipeline still reported the estimate "
                   "`speed * (1 - 0.6 * silence_fraction)`, which ignored the preset entirely. "
                   "Gap is `(reported - measured) / measured`.")
        out.append("")
        out.append(f"- mean gap: **{_mean(gaps):+.2f}%**")
        out.append(f"- worst gap: **{max(gaps, key=abs):+.2f}%**")
        out.append("")

    # ---- silence fraction, which bounds what lever 1 can deliver ----
    out.append("## Silence fraction by clip")
    out.append("")
    out.append("Lever 1's payoff scales with this. `docs/ARCHITECTURE.md` works its arithmetic "
               "from a typical podcast at s = 0.30.")
    out.append("")
    out.append("| clip | kind | silence fraction |")
    out.append("|---|---|---|")
    for clip in clips:
        vals = [r.silence_fraction for r in rows if r.clip == clip and r.condition != UNIFORM]
        kind = next((r.kind for r in rows if r.clip == clip), "")
        if vals:
            out.append(f"| {clip} | {kind} | {_mean(vals):.4f} |")
    out.append("")

    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    all_conditions = list(PRESETS.keys()) + [UNIFORM]
    p = argparse.ArgumentParser(prog="eval.harness", description=__doc__.split("\n")[0])
    p.add_argument("--clips", default="", help="comma-separated clip labels (default: all)")
    p.add_argument("--speeds", default=DEFAULT_SPEEDS,
                   help=f"comma-separated speeds (default: {DEFAULT_SPEEDS}, the band where "
                        "the ASR metric still discriminates)")
    p.add_argument("--presets", default=",".join(all_conditions),
                   help=f"comma-separated conditions (default: {','.join(all_conditions)})")
    p.add_argument("--engine", default="parakeet", help="Media API STT engine")
    p.add_argument("--no-cache", action="store_true", help="ignore cached results and re-run everything")
    args = p.parse_args(argv)

    labels = [s.strip() for s in args.clips.split(",") if s.strip()]
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    conditions = [s.strip() for s in args.presets.split(",") if s.strip()]

    bad = [c for c in conditions if c != UNIFORM and c not in PRESETS]
    if bad:
        print(f"unknown condition(s) {bad}; available: {all_conditions}", file=sys.stderr)
        return 2

    try:
        selected = manifest.by_label(labels or None)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    clips, missing = manifest.available(selected)
    for c in missing:
        print(f"note: skipping {c.label} -- source not found at {c.source}", file=sys.stderr)
    if not clips:
        print("no clips available to run", file=sys.stderr)
        return 1

    try:
        rows = run(clips, speeds, conditions, args.engine, use_cache=not args.no_cache)
    except asr.ASRUnavailable as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    RESULTS_MD.write_text(summarise(rows, speeds, conditions), encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_MD.relative_to(EVAL_DIR.parent)} and "
          f"{RESULTS_JSON.relative_to(EVAL_DIR.parent)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
