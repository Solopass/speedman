# Does Speedman actually help?

What the evaluation harness has established, what it has ruled out, and what remains
unproven. Generated from `eval/results.md` and `eval/sweep_results.md`; re-run
`python -m eval.harness` and `python -m eval.sweep` to refresh.

Measured on 12 clips of 90 s (4 non-overlapping excerpts from each of two podcasts and
one audiobook), scored by WER of parakeet on processed audio against parakeet on the
same clip at 1×.

---

## Short answer

**Yes, where it can be measured.** At 2.5×–3.5× Speedman beats a constant-rate stretch
to the same duration by 28–42%, and disabling its pause-compression lever makes results
worse on 11 of 12 clips (p = 0.006).

**At the 5×–6× product target, the mechanism verifiably operates but the benefit is
unproven.** The time map does what it claims — pauses compress ~1.9× harder than speech
at 6×, duration lands within 0.000% — but no instrument here can show that the output is
intelligible, because ASR stops discriminating above ~3.5×.

**Nothing found so far justifies changing any default.** Four parameters swept, zero
changes warranted.

---

## What is proven

### The core claim holds in the measurable band

| speed | best preset | uniform control | improvement |
|---|---|---|---|
| 2× | 0.019 | 0.022 | −12% |
| 2.5× | 0.040 (`fast`) | 0.055 | **−28%** |
| 3× | 0.188 (`fast`) | 0.326 | **−42%** |
| 3.5× | 0.543 (`aggressive`) | 0.584 | −7% |

`fast` — the shipped default — wins at both speeds where the metric discriminates best.

### Lever 1 (pause compression) earns its keep

Setting `silence_mult = 1.0`, which turns the lever off, is worse than the 2.5 default on
**11 of 12 clips (p = 0.006)**. This is the strongest positive result in the evaluation.
No tested value above 2.5 is significantly better, so the default stands.

### Duration accuracy is exact

`duration_error_pct` is 0.000% at every speed from 3× to 10×, across every clip. The
constrained water-filling solver in `ratemap.py` does what it says.

---

## What is ruled out

### `protect_window_ms` has no measurable effect between 40 ms and 160 ms

Nothing beats the 120 ms default significantly. Across four runs (2.5×/3×, n = 3 and
n = 12) the apparent winner moved between 60, 120 and 160 while the spread of means
shrank as clips were added — an effect regressing toward zero.

`ratemap.py`'s own docstring predicts this: *"Raising protection widens `p` and forces
everything else faster: wider protection is weaker protection."* Because the map is
normalised to hit the target duration exactly, widening the protected region dilutes
protection proportionally, and the two cancel. The internal decomposition changes
enormously across the sweep (onset coverage 25% → 81%) while measured intelligibility
does not move.

### `boundary_ms` shows no improvement between 100 ms and 500 ms

Nothing beats the 250 ms default. Lowest mean is the default itself.

### Raising `floor_ms` hurts

`floor_ms = 60` is worse on 9 of 10 clips (p = 0.021) and `= 100` on 11 of 12
(p = 0.006). The default of 25 is not beaten by anything.

---

## What is inert, and why that matters

`floor_ms` at 0, 10 and 25 produce **byte-identical output at 3×**. The floor never
engages there, because pauses are not yet compressed hard enough to hit it.

| speed | boundary pauses hitting the floor |
|---|---|
| 3× | 0% — inert |
| 4× | 35% |
| 5× | 61% |
| 6× | 83% |
| 8×+ | ~100% |

**The parameter that governs the target band cannot be tested by the instrument that
works below it.** The 3× sweep did not evaluate `floor_ms`; it measured a no-op. The two
values that *did* score worse (60, 100) are worse precisely because they are large enough
to bind at 3×.

This generalises: tuning questions about 5×–6× behaviour are systematically out of reach
of ASR scoring, and that gap is the single biggest hole in the evaluation.

---

## What the mechanism does in the target band

ASR cannot score 5×–6×, but the time map is inspectable at any speed. These measure that
a lever *operates*, not that it *helps*.

### Lever 1 weakens as speed rises

| speed | pause rate | speech rate | ratio | pause share of output |
|---|---|---|---|---|
| 3× | 5.77× | 2.78× | 2.07× | 7.4% |
| 5× | 9.25× | 4.65× | 1.99× | 7.7% |
| 6× | 10.60× | 5.61× | 1.89× | 8.0% |
| 10× | 14.03× | 9.55× | 1.47× | 10.0% |

As the floor engages, pauses stop compressing proportionally and begin reclaiming share
of the output. Lever 1 is fighting the floor exactly where the product needs it most.
This is a design tension, not a bug — the floor exists so boundary pauses stay audible —
but it means lever 1's contribution decays through the target band.

### Lever 2's ratio is a configuration constant

The steady-to-onset rate ratio is **2.86× at every speed from 3× to 10×**, on every clip.
That is exactly `crush_mult / protect_mult` = 2.0 / 0.7. It does not vary with content or
speed; the global normalisation guarantees it. Combined with the null result on
`protect_window_ms`, lever 2 currently has no measured effect on any outcome — only on
which audio sits in which bucket.

### No rate clamping below 10×

0 of 4289 segments hit the `max_rate = 40` ceiling anywhere from 3× to 10×. The
degeneracy noted earlier appears above ~15×, well outside the supported band. Earlier
notes putting it at 15× were measured on a single clip; on 6 clips the band through 10×
is clean.

---

## Where the assumptions are wrong

**Silence fraction.** `docs/ARCHITECTURE.md` works its arithmetic from a typical podcast
at *s* = 0.30. Measured across all 12 clips: **0.133–0.153**. Lever 1's headroom is
therefore roughly half what the architecture document claims, which flows through to the
README's "1.5–3× effective headroom" figure.

**`effective_speech_rate`.** The Studio's headline number is the formula
`speed * (1 - 0.6 * s)`, not a measurement. Against what the time map actually does it is
optimistic by a mean of 0.96% and a worst case of 4.21%, and it ignores preset entirely
though the real value varies by preset.

---

## What would close the gap

1. **A non-ASR intelligibility proxy** — STOI or a modulation-spectrum measure against the
   1× signal. Does not route through a language model, so it should survive past 3.5×, and
   it would test lever 2, which ASR may simply be blind to.
2. **A listening test.** `/api/v1/compare` already generates blind randomised sets with a
   hidden key. What is missing is a scoring protocol and somewhere to record verdicts.
   This is the only route to ground truth at 5×–6×.
3. **More clips.** n = 12 gives the sign test little power; a 2-of-12 result sits at
   p = 0.065 and cannot reach significance. Doubling the clip set would let smaller real
   effects surface.

---

## Reading the numbers

Per-clip WER ranges from 0.03 to 0.87 within a single sweep, so **differences of means are
mostly clip-difficulty noise.** Every value in a sweep is scored on the same clips, so the
comparison is paired per clip and reported as wins plus a two-sided sign-test p-value.

This is not a stylistic preference. In the `silence_mult` sweep, 3.5 had the lowest mean
WER (0.200 vs 0.224 for the default) — an apparent 11% win driven by one outlier clip. It
won on 7 of 12 clips, p = 0.774. A comparison of means would have recommended changing a
default on the strength of noise.
