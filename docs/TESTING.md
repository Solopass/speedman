# Testing & Evaluation

Everything in this project is gated on one loop: change the DSP → run the eval → did intelligibility improve? **An unvalidated measuring instrument is worse than none** — it produces confident numbers that rank noise. So this document is about the instrument first and the metric second.

---

## The central problem: you cannot ASR time-compressed audio directly

The obvious design — compress to N×, transcribe, compute WER — **does not work**, for reasons that are structural rather than fixable by picking a better model.

### CTC models have a hard output-rate ceiling

`wav2vec2-base-960h` has a feature-extractor stride of 320 samples at 16 kHz: **one frame per 20 ms, i.e. ~50 token emissions per second, maximum.** CTC cannot emit more tokens than it has frames (and needs blanks between repeated characters, so the usable rate is lower). Read speech at ~150–180 wpm is ~14–16.5 characters/second at 1x:

| speed | chars/s required | chars/s available | floor on character error |
|---|---|---|---|
| 3x | 42–50 | 50 | ~0% |
| **5x** | **70–83** | 50 | **~30–40%** |
| 8x | 112–132 | 50 | ~55–62% |
| 10x | 140–165 | 50 | ~64–70% |

**A CTC ranker saturates around 3x and is architecturally pinned above ~4x** — every variant hits the same wall and the curves *converge* instead of separating, exactly where the interesting differences are. This is not an out-of-distribution problem; it is arithmetic.

### Whisper fails differently, not better

Encoder-decoder models have no frame-rate bound on output length, but they bring a powerful internal LM that **repairs** — emitting fluent text reconstructed from context, scoring low WER on audio no human could parse — and, far out of distribution, **collapses** into repetition loops or empty output. Either failure looks like an ordinary number in a cell.

### The fix: measure by re-expansion

**Primary protocol — compress, then restore, then transcribe at 1x:**

```
original ──[variant pipeline @ N×]──> compressed
                                          │
                            [fixed canonical TSM, uniform, back to 1× length]
                                          │
                                          ▼
                                  restored ──[ASR @ 1×]──> WER vs. reference
```

This asks the right question — *what information survived the compression* — and removes the ASR frame-rate confound entirely, because every hypothesis is produced at a normal speaking rate. The re-expansion adds its own artifacts, but it adds them **identically to every variant**, so it is fair for ranking.

Rules: the restoring TSM is **fixed** for all variants and speeds (rubberband R3, uniform, no formant flag) and never co-tuned with the pipeline under test. It is a measurement fixture; changing it invalidates every historical number in `results.md`.

With re-expansion, the ranker choice stops being load-bearing. Use `wav2vec2-base-960h` as primary (weak LM prior → tracks acoustic damage rather than context-repair) and Whisper as a secondary column with greedy decoding, `condition_on_previous_text=False`, no initial prompt.

**Secondary protocol — direct ASR on the compressed audio**, reported alongside but never used for ranking above 3x. It answers a different and genuinely interesting question ("can a machine follow this as-is?") and is honest as long as the ceiling above is stated next to the numbers.

---

## Calibrating the instrument (build step 5 — before any tuning)

Three checks, all cheap, all mandatory.

**1. A graded degradation ladder.** Take one clean clip and produce a series with *known, monotonically increasing* damage — additive noise at a range of SNRs is easiest. Run it through the full harness. This establishes the harness's actual **resolution**: the smallest damage difference it reliably distinguishes.

This replaces the earlier and weaker check of "naive resample must beat plain librosa at 5x." Separating two arms that differ enormously proves nothing about resolving the 2–5 point gaps that variant tuning actually turns on — and if both arms sit near a ceiling, it proves nothing at all.

**2. Monotonicity.** WER must rise with speed for every variant. Note that this check **passes trivially** for a model pinned at 100% above its ceiling, which is why it is not sufficient on its own.

**3. Failure-mode guards.** `run_eval.py` flags cells where the hypothesis is empty, is under 20% of reference length, or repeats an n-gram more than 3 times. Flagged cells are marked in the table (e.g. `98%*`), never averaged in as data. Note that the frame-ceiling failure produces a *coherent, ~50%-length* hypothesis that trips none of these — hence check 1.

**Any speed column that fails calibration is marked unmeasurable and is not tuned against.** Expect this at the top of the range. That is information, not a blocker: it means 10–15x is evaluated by the structural metrics and listening rubric below, and the roadmap's "benchmarked, not promised" wording stands.

---

## Corpus

5–10 clips minimum, 30–60 s each, mixed voices (male/female, fast/slow, clean/podcast-quality), with **ground-truth transcripts from the corpus, never ASR output** — transcribing the original with the model you evaluate with makes the metric partly self-referential and it will flatter the pipeline.

**The corpus must span silence fractions, and LibriSpeech alone will not do.** `test-clean` is utterance-segmented read speech with silence trimmed and no disfluencies — `s ≈ 0.10`, the low extreme. By ARCHITECTURE.md's own table that caps lever 1's effect at ~6%, below the duration tolerance and inside WER noise. **The feature the project is built on would be unmeasurable on the specified corpus.** Include conversational and podcast-like material (LibriVox readings, public-domain interviews, CC-licensed podcasts) and **record each clip's measured `s` in the results table** — lever 1's benefit should correlate with it, and if it doesn't, that is a finding.

---

## Statistical discipline

5–10 clips of 30–60 s is roughly 500–1700 reference words. At 30% WER on 1000 words the binomial standard error alone is ~1.5 points. The build plan makes four sequential keep/drop decisions in the post chain, plus preset tuning, plus four lever-2 parameters — against that noise floor.

So: **`run_eval.py` reports a confidence interval on every cell, not a bare number.** Declare a minimum effect size before running (a difference smaller than it is a tie, and a tie means keep the simpler option). Grow the corpus before trusting close calls, rather than tuning harder on a small one. Without this the project will confidently lock in noise, and the eval table will make it look rigorous.

---

## Structural metrics — these work where WER doesn't

These do not depend on ASR and stay valid at 15x, which makes them the primary evidence at the top of the range.

- **Duration accuracy:** output within 2% of `input/speed`.
- **Time-map validity:** strictly monotonic; slope within `[min_rate, max_rate]`; boundary-pause floor respected; **feasibility** — the map hits target duration without violating the floor, including the degradation path when it can't (ARCHITECTURE.md).
- **Anchor precision:** with anchors every 10 ms and a local rate of 20x an output segment is ~12 samples, so ±0.5 sample rounding is ±4% local rate error and it accumulates. Test that a running residual is carried and the final anchor pinned.
- **Backend registration** — **measured; see `docs/SPIKE_RESULTS.md`.** Floor is ~3 ms mean / ~5–9 ms max output jitter, irreducible by anchor density, so protection windows must be ≥80 ms of input. `eval/spikes/registration_spike.py` is a permanent fixture: re-run it on any change of engine (R2/R3), sample rate, anchor strategy or backend, and re-run it on real speech once `align/` exists — the synthetic numbers are a lower bound.
- **Instantaneous-rate tolerance:** at what rate ratio between adjacent regions does the backend audibly break? This bounds `q(N)` (ARCHITECTURE.md) and is needed before tuning lever 2.
- **Preserved-region check** — the only *direct* measurement of lever 2. Using forced alignment on the **original**, verify transient spans get more output duration than uniform would give, and vowel-centre spans less. **This requires the `align/` module, so `align/` is v1 eval infrastructure, not Phase 1.5 work** (the feature that consumes it comes later). Without this, lever 2 has no direct measurement at all in v1.
- **Artifact guards:** no sample > 0.99, no NaN/Inf, integrated loudness within ±1 LU of target.

### On STOI

**Removed as previously specified.** It requires reference and degraded signals of the same length, time-aligned; comparing a low-speed reference against a variant at another speed satisfies neither and would return a plausible number that meant nothing.

STOI is admissible only between **two outputs at the same target speed**, aligned first (DTW or envelope cross-correlation), or the score is dominated by timing offset. Optional diagnostic, not a ranking metric. PESQ is out for the same reason.

---

## Unit tests (pytest)

- `io`: wav round-trip; mp3/m4a via ffmpeg; resample correctness; **eval path resamples 24 kHz → 16 kHz** for wav2vec2.
- `analyze`: VAD on a constructed tone-silence-tone signal; sources emit a well-formed `Annotation`.
- `ratemap`: monotonic; average honours the request; boundary-floor applied to boundary pauses **only** (short gaps compress freely — the inversion guard); infeasible-constraint case degrades as specified rather than missing duration.
- `stretch`: each backend at 2x halves duration ±2%; a sine keeps its frequency after stretching (the TSM-not-resample invariant — this would have caught the Copilot pitch confusion); `stretch_map` with constant slope agrees with `stretch`; **backends without a real time-map path raise rather than silently approximating**; staging engages only when enabled; frame scales per stage; `|y| < 1` enforced before any rubberband call.
- `post`: EQ additive (output ≈ input at gain 0); gains ramped; loudness on target; **limiter guarantees peak < 0.99 given a deliberately hot input**.
- `pipeline`: golden smoke test — bundled 5 s clip at 5x, finite, correct length.

---

## Listening checks

Spot-check at 3x and 8x: harshness (EQ overdone), phasiness/metallic ringing (PV artifacts), clicks (missing ramps), stutter, lost word boundaries (floor threshold wrong). Log in `eval/listening_notes.md` with variant + speed.

**Above the measurable ceiling, listening notes are primary evidence**, so score them on a fixed rubric — can you identify topic shifts? proper nouns? sentence boundaries? — rather than free-form impressions. Blind the variant labels; the person tuning the DSP should not know which file is theirs.

---

## CI-ability

Headless with pip-installable deps (`transformers`, `torch`, `jiwer`, `pytest`; `faster-whisper` in the optional `[eval]` extra). Keep the corpus small enough to regenerate in a few minutes on CPU. Weights download on first run — cache them and skip eval tests offline rather than failing the suite.
