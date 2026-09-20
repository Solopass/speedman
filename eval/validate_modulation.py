"""Does the modulation metric agree with WER where both work, and what does it say above?

Two parts:

1. VALIDATION. Correlate modulation statistics against cached WER at 2.5-3.5x, where the
   ASR metric still discriminates. A metric that tracks WER in the overlap has earned some
   trust above the ceiling; one that does not is measuring something else.

2. EXTRAPOLATION. Run 3x-8x for Speedman against the uniform control, where WER cannot
   follow, and see whether Speedman's predicted advantage actually shows up.

    python -m eval.validate_modulation
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
from scipy import stats

from speedman.config import build_config
from speedman.pipeline import process

from . import manifest
from .harness import CACHE_PATH, SR, UNIFORM
from .modulation import analyse

VALIDATE_SPEEDS = (2.5, 3.0, 3.5)
VALIDATE_CONDITIONS = ("natural", "fast", "aggressive", "max", UNIFORM)
EXTRAPOLATE_SPEEDS = (3.0, 4.0, 5.0, 6.0, 8.0)
N_CLIPS = 6


def render(y, speed, condition):
    uniform = condition == UNIFORM
    cfg = build_config(speed=speed, preset="fast" if uniform else condition,
                       backend="rubberband")
    if uniform:
        cfg = replace(cfg, uniform=True)
    return process(y, SR, cfg).audio


def main(argv=None) -> int:
    if not CACHE_PATH.is_file():
        print("no eval/cache.json -- run `python -m eval.harness` first", file=sys.stderr)
        return 1
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    clips, _ = manifest.available(manifest.CLIPS)
    clips = clips[:N_CLIPS]
    if not clips:
        print("no clips available", file=sys.stderr)
        return 1

    print(f"{len(clips)} clips\n")
    print("=" * 78)
    print("1. VALIDATION -- does it track WER where WER still works?")
    print("=" * 78)

    wers, fracs, peaks, centroids = [], [], [], []
    for clip in clips:
        fp = clip.fingerprint()
        y = clip.load(SR)
        for speed in VALIDATE_SPEEDS:
            for cond in VALIDATE_CONDITIONS:
                entry = cache.get(f"{fp}|{speed}|{cond}|parakeet")
                if not isinstance(entry, dict):
                    continue
                m = analyse(render(y, speed, cond), SR)
                if m.syllabic_fraction != m.syllabic_fraction:
                    continue
                wers.append(entry["wer"])
                fracs.append(m.syllabic_fraction)
                peaks.append(m.peak_hz)
                centroids.append(m.centroid_hz)

    if len(wers) < 8:
        print(f"  only {len(wers)} paired points -- run the harness first")
        return 1

    print(f"  n = {len(wers)} paired (WER, modulation) observations\n")
    print(f"  {'statistic':<22} {'Spearman rho vs WER':>21} {'p':>10}")
    print("  " + "-" * 55)
    for name, values, expect in (
        ("syllabic_fraction", fracs, "negative"),
        ("peak_hz", peaks, "positive"),
        ("centroid_hz", centroids, "positive"),
    ):
        rho, p = stats.spearmanr(values, wers)
        ok = "as predicted" if (rho < 0) == (expect == "negative") and p < 0.05 else \
             ("not significant" if p >= 0.05 else "WRONG DIRECTION")
        print(f"  {name:<22} {rho:>+21.3f} {p:>10.4f}   {ok}")

    print()
    print("  Predicted signs: more syllabic-band energy -> lower WER (negative rho);")
    print("  a higher modulation peak or centroid means faster-than-syllabic structure,")
    print("  which should raise WER (positive rho).")

    print()
    print("=" * 78)
    print("2. EXTRAPOLATION -- what it says above the ASR ceiling")
    print("=" * 78)
    print(f"  {'speed':>6}  {'syllabic fraction':>28}   {'modulation peak (Hz)':>30}")
    print(f"  {'':>6}  {'speedman':>12} {'uniform':>8} {'gain':>6}   "
          f"{'speedman':>12} {'uniform':>8} {'ratio':>7}")
    print("  " + "-" * 74)

    rows = []
    for speed in EXTRAPOLATE_SPEEDS:
        sm_f, un_f, sm_p, un_p = [], [], [], []
        for clip in clips:
            y = clip.load(SR)
            a = analyse(render(y, speed, "fast"), SR)
            b = analyse(render(y, speed, UNIFORM), SR)
            sm_f.append(a.syllabic_fraction); sm_p.append(a.peak_hz)
            un_f.append(b.syllabic_fraction); un_p.append(b.peak_hz)

        sf, uf = float(np.mean(sm_f)), float(np.mean(un_f))
        sp_, up = float(np.mean(sm_p)), float(np.mean(un_p))
        wins = sum(1 for x, y_ in zip(sm_f, un_f) if x > y_)
        rows.append((speed, sf, uf, sp_, up, wins, len(sm_f)))
        print(f"  {speed:>5.1f}x  {sf:>12.3f} {uf:>8.3f} {(sf-uf)/uf*100:>+5.1f}%   "
              f"{sp_:>12.2f} {up:>8.2f} {up/sp_:>6.2f}x")

    print()
    total_wins = sum(r[5] for r in rows)
    total_n = sum(r[6] for r in rows)
    print(f"  Speedman retains more syllabic-band energy than uniform on "
          f"{total_wins}/{total_n} clip-speed pairs.")
    print()
    print("  Read this as a correlate, not a verdict: it cannot say 6x is intelligible,")
    print("  only whether Speedman preserves more of the structure that carries speech.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
