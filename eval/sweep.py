"""Sweep one DSP parameter and score each value with the same WER metric as the harness.

    python -m eval.sweep --param protect_window_ms --values 80,100,120,160
    python -m eval.sweep --param crush_mult --values 1.5,2.0,2.5 --speed 3

Runs at 3x by default: `eval/README.md` shows that is where the ASR metric discriminates
best (at 2x everything is near the floor, above 3.5x everything saturates). A tuning
answer from a saturated speed is noise, so the default is the speed that can actually
settle the question.

Reference transcripts are shared with `eval.harness` through `eval/cache.json`, so a
sweep after a full harness run reuses the 1x references instead of re-transcribing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from speedman import analyze, io as sio, ratemap
from speedman.analyze import SpanKind
from speedman.config import PRESETS, RateMapConfig, build_config
from speedman.pipeline import process

from . import asr, manifest
from .harness import CACHE_PATH, SR, WORK_DIR, UNIFORM, _cache_load, _cache_save
from .wer import wer, cer

SWEEP_MD = Path(__file__).resolve().parent / "sweep_results.md"

RATEMAP_FIELDS = {f.name for f in RateMapConfig.__dataclass_fields__.values()}


def decompose(ann) -> dict[str, float]:
    """Percentage of the signal in each span kind -- the diagnostic that explains
    *why* a parameter value scores the way it does."""
    totals: dict[str, int] = {}
    for sp in ann.spans:
        totals[sp.kind.value] = totals.get(sp.kind.value, 0) + sp.length
    n = sum(totals.values()) or 1
    return {k: 100.0 * v / n for k, v in totals.items()}


def analyse(y: np.ndarray, cfg, speed: float) -> dict:
    """Annotation decomposition plus the rates the time map actually applies."""
    ann = analyze.annotate_vad(
        y, SR,
        protect_window_ms=cfg.ratemap.protect_window_ms,
        protect_lead_ms=cfg.ratemap.protect_lead_ms,
    )
    tm = ratemap.build_time_map(ann, speed, cfg.ratemap)
    lengths = np.diff(tm.seg_bounds).astype(float)
    out_lens = lengths / np.maximum(tm.rates, 1e-9)
    kinds = np.array([s.kind.value for s in ann.spans])

    def rate_for(mask) -> float:
        if not mask.any():
            return float("nan")
        return float(lengths[mask].sum() / max(out_lens[mask].sum(), 1e-9))

    d = decompose(ann)
    return {
        "onset_pct": d.get("onset", 0.0),
        "steady_pct": d.get("steady", 0.0),
        "silence_pct": d.get("silence", 0.0),
        "onset_rate": rate_for(kinds == "onset"),
        "steady_rate": rate_for(kinds == "steady"),
        "speech_rate": rate_for(kinds != "silence"),
        "onsets_per_s": ann.meta.get("n_onsets", 0) / (len(y) / SR),
    }


def build(preset: str, speed: float, param: str | None, value: float | None):
    """Base preset with one ratemap field overridden."""
    cfg = build_config(speed=speed, preset=preset, backend="rubberband")
    if param is None:
        return cfg
    return replace(cfg, ratemap=replace(cfg.ratemap, **{param: value}))


def score(y, clip, speed, cfg, cache_key, ref_text, cache, engine, out_name) -> dict:
    if cache_key in cache:
        return cache[cache_key]

    out_path = WORK_DIR / out_name
    t0 = time.perf_counter()
    res = process(y, SR, cfg)
    elapsed = time.perf_counter() - t0
    sio.save(out_path, res.audio, res.sr)

    hyp = asr.transcribe(out_path, engine=engine).text
    w, c = wer(ref_text, hyp), cer(ref_text, hyp)
    entry = {"wer": w.rate, "cer": c.rate, "process_s": elapsed}
    cache[cache_key] = entry
    _cache_save(cache)
    return entry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval.sweep", description=__doc__.split("\n")[0])
    p.add_argument("--param", required=True, help=f"RateMapConfig field: {sorted(RATEMAP_FIELDS)}")
    p.add_argument("--values", required=True, help="comma-separated values to try")
    p.add_argument("--speed", type=float, default=3.0, help="speed to evaluate at (default: 3.0)")
    p.add_argument("--preset", default="fast", help="base preset (default: fast)")
    p.add_argument("--clips", default="", help="comma-separated clip labels (default: all)")
    p.add_argument("--engine", default="parakeet")
    args = p.parse_args(argv)

    if args.param not in RATEMAP_FIELDS:
        print(f"unknown param {args.param!r}; RateMapConfig has {sorted(RATEMAP_FIELDS)}", file=sys.stderr)
        return 2
    if args.preset not in PRESETS:
        print(f"unknown preset {args.preset!r}; available: {sorted(PRESETS)}", file=sys.stderr)
        return 2

    values = [float(v) for v in args.values.split(",") if v.strip()]
    baseline = getattr(RateMapConfig(), args.param)

    try:
        selected = manifest.by_label([s.strip() for s in args.clips.split(",") if s.strip()] or None)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    clips, missing = manifest.available(selected)
    for c in missing:
        print(f"note: skipping {c.label} -- source not found", file=sys.stderr)
    if not clips:
        print("no clips available", file=sys.stderr)
        return 1

    try:
        asr.require_online()
    except asr.ASRUnavailable as e:
        print(e, file=sys.stderr)
        return 1

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_load(True)

    print(f"sweeping {args.param} over {values} at {args.speed:g}x, base preset '{args.preset}'")
    print(f"(preset default for this field: {baseline})\n")

    results: dict[float, dict] = {}
    diagnostics: dict[float, dict] = {}
    uniform_wers: list[float] = []

    for clip in clips:
        fp = clip.fingerprint()
        ref_key = f"{fp}|ref|{args.engine}"
        if ref_key not in cache:
            print(f"[{clip.label}] transcribing 1x reference...", flush=True)
            y0 = clip.load(SR)
            ref_path = WORK_DIR / f"{clip.label}_{fp}_1x.wav"
            sio.save(ref_path, y0, SR)
            cache[ref_key] = asr.transcribe(ref_path, engine=args.engine).text
            _cache_save(cache)
        ref_text = cache[ref_key]
        if not ref_text.strip():
            print(f"[{clip.label}] SKIP: empty reference")
            continue

        print(f"[{clip.label}]", flush=True)
        y = clip.load(SR)

        # Uniform control at this speed, for context on every row.
        uni_cfg = replace(build_config(speed=args.speed, preset=args.preset,
                                       backend="rubberband"), uniform=True)
        uni = score(y, clip, args.speed, uni_cfg,
                    f"{fp}|{args.speed}|{UNIFORM}|{args.engine}", ref_text, cache,
                    args.engine, f"{clip.label}_{fp}_{args.speed:g}x_uniform.wav")
        uniform_wers.append(uni["wer"])
        print(f"  {'uniform control':<26} WER {uni['wer']:.3f}")

        for v in values:
            cfg = build(args.preset, args.speed, args.param, v)
            key = f"{fp}|{args.speed}|{args.preset}|{args.param}={v}|{args.engine}"
            name = f"{clip.label}_{fp}_{args.speed:g}x_{args.preset}_{args.param}{v:g}.wav"
            entry = score(y, clip, args.speed, cfg, key, ref_text, cache, args.engine, name)
            results.setdefault(v, {})[clip.label] = entry["wer"]

            if v not in diagnostics:
                diagnostics[v] = analyse(y, cfg, args.speed)
            mark = "  <- preset default" if v == baseline else ""
            print(f"  {args.param}={v:<14g} WER {entry['wer']:.3f}{mark}")

    if not results:
        print("nothing scored", file=sys.stderr)
        return 1

    lines = _report(args, values, results, diagnostics, uniform_wers, baseline, clips)
    SWEEP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {SWEEP_MD.relative_to(SWEEP_MD.parent.parent)}")
    return 0


def _sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p under H0: a value is no better than the default.

    The right test for this design: per-clip WER varies enormously (0.03 to 0.87 in the
    same sweep), so a difference of means is mostly clip-difficulty noise. Pairing on the
    clip removes that, and the sign test needs no assumption about the shape of what is
    left -- which matters, because WER is bounded, skewed, and nothing like normal.
    """
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _paired(value_per_clip: dict, base_per_clip: dict) -> str:
    """'wins/n (p=...)' for one value against the default, paired by clip."""
    shared = [c for c in value_per_clip if c in base_per_clip]
    if not shared or value_per_clip is base_per_clip:
        return "--"
    wins = sum(1 for c in shared if value_per_clip[c] < base_per_clip[c])
    ties = sum(1 for c in shared if value_per_clip[c] == base_per_clip[c])
    n = len(shared) - ties
    if n == 0:
        return "identical"
    p = _sign_test_p(wins, n)
    star = " **✓**" if p < 0.05 and wins > n - wins else ""
    return f"{wins}/{n} (p={p:.3f}){star}"


def _report(args, values, results, diagnostics, uniform_wers, baseline, clips) -> list[str]:
    def mean(vs):
        vs = [v for v in vs if v == v]
        return statistics.fmean(vs) if vs else float("nan")

    uni = mean(uniform_wers)
    out = [
        f"# Sweep: `{args.param}` at {args.speed:g}x",
        "",
        f"Generated by `python -m eval.sweep` on {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        f"Base preset `{args.preset}`, {len(clips)} clip(s). Uniform control at this speed: "
        f"**WER {uni:.3f}**. Preset default for this field: `{baseline}`.",
        "",
        "Every value is scored on the same clips, so `beats default` is the comparison that "
        "counts: it pairs per clip, which cancels the large differences in clip difficulty "
        "that swamp a comparison of means.",
        "",
        "| value | mean WER | vs default | vs uniform | beats default | onset% | steady% | onset rate | steady rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    base_wer = mean(results.get(baseline, {}).values()) if baseline in results else float("nan")
    base_per_clip = results.get(baseline, {})
    for v in values:
        m = mean(results.get(v, {}).values())
        d = diagnostics.get(v, {})
        vs_def = f"{(m - base_wer) / base_wer * 100:+.0f}%" if base_wer == base_wer and base_wer else "--"
        vs_uni = f"{(m - uni) / uni * 100:+.0f}%" if uni == uni and uni else "--"
        tag = " **(default)**" if v == baseline else ""
        out.append(
            f"| {v:g}{tag} | {m:.3f} | {vs_def} | {vs_uni} | {_paired(results.get(v, {}), base_per_clip)} | "
            f"{d.get('onset_pct', float('nan')):.1f}% | {d.get('steady_pct', float('nan')):.1f}% | "
            f"{d.get('onset_rate', float('nan')):.2f}x | {d.get('steady_rate', float('nan')):.2f}x |"
        )

    out += ["", "## Per-clip", "", "| value | " + " | ".join(c.label for c in clips) + " |",
            "|" + "---|" * (len(clips) + 1)]
    for v in values:
        row = [f"{results.get(v, {}).get(c.label, float('nan')):.3f}" for c in clips]
        out.append(f"| {v:g} | " + " | ".join(row) + " |")

    # A winner only counts if it beats the default on significantly more clips than not.
    significant = []
    for v in values:
        if v == baseline:
            continue
        per_clip = results.get(v, {})
        shared = [c for c in per_clip if c in base_per_clip]
        wins = sum(1 for c in shared if per_clip[c] < base_per_clip[c])
        n = sum(1 for c in shared if per_clip[c] != base_per_clip[c])
        if n and _sign_test_p(wins, n) < 0.05 and wins > n - wins:
            significant.append((v, mean(per_clip.values()), wins, n))

    out.append("")
    if significant:
        v, m, wins, n = min(significant, key=lambda t: t[1])
        out.append(f"**Change the default: `{args.param} = {v:g}`** beats the current "
                   f"`{baseline:g}` on {wins} of {n} clips (WER {m:.3f} vs {base_wer:.3f}).")
    else:
        lo = min(values, key=lambda v: mean(results.get(v, {}).values()))
        out.append(f"**No change warranted.** The lowest mean WER is `{args.param} = {lo:g}` "
                   f"({mean(results[lo].values()):.3f} vs {base_wer:.3f} for the default), but no "
                   f"value beats the default on significantly more clips than it loses to it. "
                   f"Differences of means at this sample size are clip-difficulty noise.")
    if len(clips) < 8:
        out.append("")
        out.append(f"> Only {len(clips)} clips: the sign test has little power here. "
                   "Add clips to `manifest.py` before trusting a null result.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
