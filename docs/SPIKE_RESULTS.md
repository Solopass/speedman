# Spike results — time-map registration precision

Run 2026-08-27, before any implementation, to answer the one question that could have forced an architecture rewrite: **can rubberband place a non-uniform time map precisely enough for transient protection to work at all?**

Script: `eval/spikes/registration_spike.py`. Environment: rubberband CLI 3.3.0, pyrubberband 0.4.0, 24 kHz mono. Synthetic signal with alternating 0.8 s speech / 0.2 s pause blocks and a 4 ms 6 kHz marker burst at a known offset in each block; a full two-lever rate profile (transients 0.7×, vowel centres 2.0×, pauses 2.5×) normalized to the target speed; markers recovered from the output by bandpass + envelope peak-picking and compared against the position the supplied anchors asked for.

## Results

| N | anchor spacing | mean abs error | max abs error | detrended jitter |
|---|---|---|---|---|
| 5x | 20 ms | 7.15 ms | 13.09 ms | 3.79 ms |
| 5x | **10 ms** | **3.64 ms** | 8.83 ms | 4.17 ms |
| 5x | 5 ms | 3.48 ms | 8.83 ms | 4.23 ms |
| 5x | 2 ms | 4.07 ms | 8.75 ms | 4.40 ms |
| 8x | 20 ms | 5.70 ms | 9.95 ms | 4.55 ms |
| 8x | **10 ms** | **2.42 ms** | 4.47 ms | 2.32 ms |
| 8x | 5 ms | 2.57 ms | 4.48 ms | 2.31 ms |
| 10x | 20 ms | 4.46 ms | 7.25 ms | 2.56 ms |
| 10x | **10 ms** | **3.08 ms** | 5.75 ms | 3.53 ms |
| 10x | 2 ms | 3.17 ms | 5.62 ms | 3.58 ms |

**Duration accuracy: 0.00% error at every speed and every anchor spacing**, given a running residual and a pinned final anchor.

## What this establishes

**1. Duration control is solved.** Exact output length, non-uniform map, no tricks beyond pinning the last anchor and forcing strict monotonicity. The ±2% tolerance is not in danger.

**2. Anchor spacing has an optimum at ~10 ms, and finer is wasted.** 20 ms → 10 ms roughly halves the error; 10 → 5 → 2 ms is flat. The residual is *rubberband's internal windowing*, not our sampling of the map. **Use 10 ms anchors.** For a 10-hour audiobook that is still 3.6M anchors and a ~50 MB timemap file — another reason chunking is required, not optional.

**3. The residual is jitter, not drift.** Detrending against file position leaves 2.3–4.4 ms of scatter, and the drift slope is small and inconsistent in sign. **This error cannot be corrected by better anchoring, residual carrying, or feedback.** It is a floor: **~3 ms mean, ~5–9 ms worst case, in output time.**

## The consequence that changes the design

A protection window must be substantially longer than the placement jitter, or the protection lands on the wrong audio.

At 10x with the profile above, the normalized transient rate is ~5.5x, so a window of `W` ms of input occupies `W/5.5` ms of output. For the window's output extent to be ≥3× the ~5 ms worst-case jitter, it needs ≥15 ms of output — i.e. **≥ ~80 ms of input, and ~150 ms to be comfortable.**

**Therefore: protection operates at syllable-onset / word-onset granularity, not phoneme granularity.** ARCHITECTURE.md's illustration of protecting a specific 60 ms VOT is not mechanically achievable — you cannot place a 60 ms window to ±5 ms. What *is* achievable is protecting the ~150 ms syllable onset that contains the burst. The burst still gets a slower rate; the targeting is simply coarser.

Two follow-on effects:

- **Wider windows cost more budget.** The protected fraction `p` rises (0.15 → ~0.30), and the normalization bracket `v/m + p/q + (1−v−p)` then exceeds 1, meaning everything else must run faster to compensate. **Wider protection means weaker protection.** `p` and `q` trade against each other and must be tuned jointly, not independently.
- **Phoneme-level annotation is wasted precision.** See below.

## Scope cut: Tier 2 phoneme alignment is not worth building

TRANSCRIPT_ALIGNMENT.md proposed a Tier 2 using the Montreal Forced Aligner to get true phoneme boundaries at ~10 ms precision, at the cost of a large external dependency.

**That precision cannot be exploited.** The mechanism places windows to ±5 ms and needs them ≥80 ms wide. Word-level alignment from `torchaudio.functional.forced_align` — 20 ms quantum, no new heavy dependency — is *already at or beyond* the useful precision of the thing consuming it.

**Recommendation: drop MFA from the roadmap.** Build Tier 1 only. Revisit only if the backend's registration floor improves by an order of magnitude, which would require a different stretching engine, not a different aligner.

## Caveats

- Synthetic signal, not real speech. The marker is a clean 6 kHz burst — deliberately easy to localise. Real consonant bursts are broadband and harder to place *and* to measure, so **treat these numbers as a lower bound on the error.** Re-run on real speech with aligned ground truth once `align/` exists.
- Rubberband R2 (the default engine). R3 may register differently; re-run with `{'-3': ''}` before choosing an engine.
- Single 10 s file, 10 markers per condition. Enough to establish the order of magnitude and that the error is jitter rather than drift; not enough for a precise floor.

## What to re-run and when

This is a fixture, not a one-off. Re-run `eval/spikes/registration_spike.py` when changing engine (R2/R3), sample rate, anchor strategy, or backend. Registration error is a property of the stretching engine, and every lever-2 parameter is downstream of it.
