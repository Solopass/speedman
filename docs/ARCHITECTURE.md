# Architecture — v1 pipeline

## Design principles

**Compress time, not information.** Intelligibility at speed depends on things the pipeline exists to protect: formants (vowel identity), consonant transients (onset bursts — where most word-recognition information lives), prosodic boundaries, pitch contour, and spectral envelope.

**Not all audio deserves equal compression.** True silence between phrases carries little linguistic information — compress it hardest. Steady vowel centres are redundant — compress them hard. Consonant transients are dense — compress them least. (Note the qualifier: *some* near-silence is phonemically critical. See "the stop-closure trap" below.)

**Measure, don't guess.** Every DSP change is validated against the eval harness (TESTING.md). No stage ships because it sounds plausible — and that includes the claims in this document.

---

## What non-uniform compression is actually worth

This is the project's headline feature, so its ceiling gets computed rather than assumed. There are **two distinct levers** working by different mechanisms. Conflating them is how the feature gets oversold.

### Lever 1 — pause compression: a genuine reduction in the speech rate

Let `s` = silence as a fraction of original duration, `k` = how much harder silence is compressed (design value 2.5), `N` = requested speed, `r` = the rate the speech itself experiences.

```
output = s·T/(k·r) + (1−s)·T/r = T/N
  ⟹  r = N · (1 − s(1 − 1/k))      # = N(1 − 0.6s) at k = 2.5
```

| silence fraction `s` | typical material | speech experiences |
|---|---|---|
| 0.10 | tightly edited audiobook, LibriSpeech | 0.94 N |
| 0.20 | podcast, lecture | 0.88 N |
| 0.35 | conversational, interview | 0.79 N |
| 0.50 | slow speaker, heavy pausing | 0.70 N |

At a requested 5x on podcast material the speech is really stretched at 4.4x. Real, worth having, **not** a transformation. The win scales with the source's silence, so the CLI reports measured `s` at `--verbose` — it predicts how well a given file will do.

### Lever 1 saturates — and then it *inverts*

The `≥ F` boundary floor and the honour-the-average constraint fight each other. A pause of length `P` cannot be compressed past `P/F` without breaching the floor. With `F = 25 ms`:

- **Saturation:** requested silence rate is `k·r ≈ 2.2N`; for `P = 300 ms` the cap is 12x, so the floor binds above `N ≈ 5.5x`. At 10x with `s = 0.20` the speech ends at **9.6x instead of 10x** — a 4% win. The feature has all but evaporated.
- **Inversion (this is the part that matters):** once `N > P/F`, floored pauses consume *more* than their uniform share, and the speech is pushed **faster than requested**.

| typical pause `P` | inverts above | speech rate at requested 10x, s=0.20 |
|---|---|---|
| 500 ms | 20x | 8.9x — win |
| 300 ms | 12x | 9.6x — marginal |
| 200 ms | 8x | **10.7x — loss** |
| 150 ms | 6x | **12.0x — loss** |
| 100 ms | 4x | **16.0x — loss** |

Inter-*word* gaps in running speech are ~100–200 ms and far more numerous than the 300 ms phrase boundaries the saturation example used. **Applying the floor to every detected silence makes lever 1 net-harmful somewhere inside the 4–8x range — the daily-driver band.**

**Therefore: the floor applies only to boundary pauses, not to every gap.** `ratemap.py` classifies silences by original duration; only those above a threshold (`--boundary-ms`, default ~250 ms — tune it) are floored and treated as prosodic boundaries. Short inter-word gaps compress freely with the speech. Saturation is acceptable behaviour; inversion is a bug, and this is the fix.

**Feasibility must be checked explicitly.** Floored pauses owe `F × n_boundaries` of output time, and that can exceed the entire target duration (a 60 s clip with 200 boundaries owes 5 s of floor; at 15x the whole target is 4 s). `ratemap.py` detects infeasibility and degrades in a defined order — raise the boundary threshold, then relax `F` — with a `--verbose` warning. It must never silently miss the duration target.

### Lever 2 — within-speech reallocation

**Lever 2 cannot reduce the average rate, by construction.** The rate map is normalized to hit the requested duration, so whatever is gained by crushing vowel centres is handed straight back protecting transients. This is a design constraint, not an empirical finding — do not expect the eval table to show lever 2 producing a shorter file.

What it does is **reallocate a fixed time budget from redundant regions to critical ones.** The output is the same length; the consonants inside it live in a slower regime. With protection factor `q`, a 60 ms stop-consonant VOT survives as:

| speed | uniform | protected at q=0.7 |
|---|---|---|
| 5x | 12.0 ms | 17.1 ms |
| 8x | 7.5 ms | 10.7 ms |
| 10x | 6.0 ms | 8.6 ms |
| 15x | 4.0 ms | 5.7 ms |

**Measured constraint (docs/SPIKE_RESULTS.md): protection cannot be surgical.** Rubberband places time-map key frames with ~3 ms mean / ~5–9 ms worst-case jitter, and that floor is *not* reducible by finer anchors. A protection window must be ≥ ~80 ms of input (≥150 ms to be comfortable) or it lands on the wrong audio. **So protection happens at syllable-onset granularity, not phoneme granularity** — the 60 ms VOT above is an illustration of the physics, not a window you can actually place. The burst still gets a slower rate as part of a wider onset region. Note also that wider windows raise `p`, which by normalization forces everything else faster: **wider protection is weaker protection**, so `p` and `q` must be tuned jointly.

**Lever 2 is scale-invariant, not unbounded.** `q` multiplies `N`, so protected duration still falls as `1/N`. At fixed `q = 0.7` it moves the wall from ~6x to ~8.6x — a real 43% gain, and *gone well before 15x*. Holding a *fixed absolute* VOT instead means shrinking `q` with `N`, and the cost explodes:

| N | `q` to hold 15 ms | remainder rate | instantaneous rate ratio |
|---|---|---|---|
| 8x | 0.50 | 8.7x | 2.2 : 1 |
| 10x | 0.40 | 12.4x | 3.1 : 1 |
| 15x | 0.27 | **28.9x** | **7.2 : 1** |

A 7:1 swing in instantaneous rate across tens of milliseconds is not something WSOLA or a phase vocoder absorbs gracefully. **So `q` is a function of `N`, not a constant, and the achievable protection is bounded by what the backend tolerates.** Measuring that tolerance — at what instantaneous rate ratio the backend audibly breaks — is a v1 eval row in its own right.

### The premise underneath lever 2 is a hypothesis, not physics

Lever 2's justification is that below some duration a consonant burst stops doing its job. **The mechanism is contested and the docs previously overstated it.** Peripheral auditory temporal resolution is far finer than the durations here (gap-detection thresholds are ~2–3 ms), so "the ear cannot resolve a 12 ms burst" is *wrong*. The classical time-compressed-speech literature (Foulke & Sticht and successors) locates comprehension breakdown around 275–300 wpm and attributes it to **central processing capacity** — the rate at which linguistic units can be identified and integrated — not to peripheral resolution.

This matters practically. If the bottleneck is central rather than peripheral, protecting a burst's *duration* buys less than assumed, because the listener's problem is arrival rate, not physical resolvability. **Lever 2 therefore ships as a measured hypothesis.** If the eval table shows reallocation not helping, that is a real result and the project should believe it rather than tune around it. Someone should also read the primary literature and replace this paragraph with sourced numbers.

### Consequence for the build

Lever 1 is bounded, saturates ~5.5x, and inverts if the floor is applied naively. Lever 2 is duration-neutral, scale-invariant, and bounded by backend tolerance. **Neither is a magic bullet, and together they buy perhaps 1.5–3x of effective headroom, not 10x.** Both ship in v1 because both are cheap and the combination is the best available; the roadmap's honest framing (PLAN.md) stands, and nothing here supports promising 15x.

---

## v1 pipeline flow

```
load (ffmpeg → mono float32 @ 24 kHz)
  → analyze: VAD segments + onset strength      [on ORIGINAL audio]
  → build a global TIME MAP: monotonic input→output sample mapping,
      local slope = instantaneous rate
        boundary pauses (> ~250 ms): rate × ~2.5, floored at ~25 ms
        short inter-word gaps:       compress with the speech (NOT floored)
        transients:                  rate × q(N)   (protected)
        vowel steady-state:          rate × ~2.0   (crushed)
        remainder:                   normalized to hit the requested duration
  → apply the time map in ONE pass (staged if enabled and rate > 4)
  → post: presence EQ (additive peaking, 2–5 kHz, ≤ +4 dB)
        → optional transient enhancement (ramped, onset-gated)
        → gentle dynamic-range compression
        → EBU R128 normalize to −16 LUFS
        → LIMITER (required — see below)
  → write output
```

Analysis runs on the original audio. VAD and onset detection are reliable at 1x and unreliable on compressed audio; the Copilot draft ran them post-compression (see code review).

---

## Key decisions

### One global time map, not per-segment stretch-and-join

The most important structural decision in v1.

- *Correctness:* segment-wise stretching needs a crossfade at every join. At 10x a 10 ms crossfade spans most of a compressed phoneme — smearing exactly the boundaries lever 2 exists to protect. A time map has no joins.
- *Performance:* `pyrubberband` shells out to the CLI and round-trips a temp WAV **per call**. Per-segment stretching means hundreds of subprocess spawns per file; on Windows that alone could exceed the listening time of the source.

**Verified:** `pyrubberband.timemap_stretch(y, sr, time_map, rbargs=None)` exists (0.4.0) and works; `rubberband` CLI has `-M/--timemap` (v3.3.0). Measured: exact duration on a 60 s / 6001-anchor map in ~0.4 s wall.

### Backend reality — read before designing `stretch.py`

The interface exposes `stretch(y, rate)` and `stretch_map(y, time_map)`, but **only one backend can actually do the second one:**

| backend | constant rate | time map |
|---|---|---|
| `rubberband` | yes | **yes** — the only real one |
| `phasevocoder` (librosa) | yes | **no** — `time_stretch(rate=…)` is scalar-only; passing an array raises |
| `wsola` (audiotsm) | yes | **no** — see below |
| `resample` | yes | meaningless — a varying rate produces continuous pitch warble |

**`audiotsm` does not support a variable hop, contrary to earlier drafts of this document.** Measured on 0.1.2: `set_speed` mutates only `analysis_hop`; `frame_length` and `synthesis_hop` are fixed at construction. Worse, when `analysis_hop > frame_length` the library *silently discards* input — 60% of samples dropped at 5x, 80% at 10x with defaults, deleting any consonant landing in a skipped span. Even configured correctly it misses the ±2% duration tolerance above 2x, and its rate can only change once per synthesis hop (~85 ms of input at 10x) — coarser than the 60 ms events lever 2 targets.

**Consequences, which the plan must accept rather than paper over:**
1. **`rubberband-cli` is a hard dependency for the headline feature**, not an optional accelerator. Without it the pipeline can still do uniform stretching, but non-uniform compression is unavailable. Say so in the install docs and fail loudly, not silently.
2. The three non-rubberband backends are **eval baselines and controls**, not fallbacks. Do not present them as a degradation path.
3. If a pure-Python time-map path is genuinely needed later, it means hand-rolling variable-rate WSOLA. That is a real project, not a fallback — scope it separately.

### `--formant` does nothing here

Rubberband's `-F/--formant` is documented as *"formant preservation when pitch shifting"*. This pipeline never pitch-shifts, so the flag is inert. Earlier drafts cited it in four places as a rubberband advantage; that was wrong. **Formants are preserved by any TSM that isn't resampling** — which is the actual reason TSM beats resampling, stated below. Keep `--formant` off and stop advertising it.

Also: `pyrubberband` invokes the binary as `rubberband`, whose default engine is R2. R3 requires passing `{'-3': ''}` in `rbargs`, and costs significantly more CPU. Decide deliberately and cost it against the 3-hour-file budget.

### Working sample rate: 24 kHz mono

Speech lives below 12 kHz; halving the rate quarters STFT cost. **The eval path must resample to 16 kHz** for any wav2vec2-family model — easy to get silently wrong.

### Staged stretching is a measurable flag

`--staged/--no-staged`; `rate > 4 → ⌈log₄(rate)⌉ passes`. The claim that staged 10x beats single-pass 10x is plausible but unproven, so it ships as a row in `eval/results.md` and defaults on only if it wins.

Two unsolved details, to be specified before implementing: (a) `⌈log₄(rate)⌉` is defined for a *scalar* rate and is undefined for a time map — decomposing a monotonic map into two maps of "equal" slope is a real design problem; (b) analysis happens on the original, so stage 2's map lives in stage-1 *output* coordinates, requiring every annotation boundary to be pushed through stage 1's **realized** (not nominal) mapping. **If this cannot be specified cleanly, ship staging for constant-rate mode only and say so.**

### Frame size must scale per stage

After a 3.16x pass the audio is 3.16x denser in transients. A frame held constant across stages spans several phonemes on pass 2 and smears what pass 1 preserved. Most staging disappointments are this bug, not the staging idea.

### A limiter is required in v1

The post chain is additive EQ (+4 dB) → transient boost → DRC → normalize to −16 LUFS. Speech at −16 LUFS with a 12–18 dB crest factor already peaks near 0 dBFS *before* those boosts. **Without a limiter the chain cannot satisfy its own "no sample > 0.99" criterion.** A limiter belongs at the end of v1's chain, not only in v3's real-time notes.

Separately: `pyrubberband` writes its temp WAV as PCM_16, so every `timemap_stretch` call is a silent 16-bit quantize-and-clip. **Guarantee `|y| < 1` before the stretch**, and note the precision floor.

### Chunking vs. the global map — an unresolved tension

A 3-hour file is ~1 GB at 24 kHz float32, and `timemap_stretch` requires the *entire* signal in memory plus a temp WAV plus the output. But chunking reintroduces per-chunk subprocess spawns and per-chunk seams — rubberband's window state does not survive across process invocations — which is precisely what the global map was adopted to avoid.

**Audiobooks are a primary target, so ten-hour files are the normal case** — ~3.5 GB at 24 kHz float32. Whole-file processing is therefore not an acceptable answer, and this is a v1 requirement rather than a deferred question.

**Solve it as (b): chunk at long boundary pauses only.** A seam placed inside a >250 ms silence, at a point where the map is locally uniform, is the least damaging cut available — and boundary pauses are already being detected for lever 1, so the information is free. Validate by processing a shorter file both chunked and unchunked and measuring the difference; if the seams are audible or measurable, the design needs revisiting before v1 ships.

### Unchanged decisions

**No pre-emphasis in the output path.** An analysis trick; applied to output without de-emphasis it just makes everything harsh (the Copilot draft did this).

**No pitch "normalization" stage.** TSM does not change pitch — that is the entire point of TSM over resampling. There is no chipmunk effect to correct.

**Pauses are compressed, not fabricated.** Real pauses kept short beat synthetic zero-insertion, which clicks and lands in the wrong places.

---

## Annotation sources

The time map needs to know which samples are boundary pause, transient and steady-state. `analyze.py` gets that from a pluggable source:

| Source | Availability | Quality |
|---|---|---|
| `vad` (default) | always | VAD segments + onset strength — heuristic, meaningful error rate |
| `aligned` | when a transcript exists | forced-alignment boundaries — **better, but still a model output with an error rate, not ground truth** |

Both emit the same `Annotation` structure. v1 builds `vad`; `aligned` is designed in `docs/TRANSCRIPT_ALIGNMENT.md`. **Define the interface in v1 even with one implementation** — retrofitting a second source into rate-map code written for one is a rewrite.

### The stop-closure trap

The ~50–80 ms of near-silence during a stop consonant is not a pause — it is part of the phoneme. VAD reads it as silence, and lever 1 crushes it while lever 2 is simultaneously trying to protect the burst that follows. **The two levers fight each other, and no DSP tuning fixes it, because the error is in the annotation.** In v1 the mitigation is partial: the boundary-duration threshold (~250 ms) excludes most closures from flooring by construction. A real fix needs phoneme-level annotation — see TRANSCRIPT_ALIGNMENT.md, and note the caveats there about what is actually achievable.

**Package layout**

```
src/speedman/
  io.py          # load/save, ffmpeg wrapper, resampling
  analyze.py     # annotation sources (vad | aligned) → Annotation
  ratemap.py     # Annotation → global time map; normalization, floors, feasibility
  stretch.py     # TimeStretcher interface + backends + staged chaining
  post.py        # EQ, transient enhancement, DRC, loudness, limiter
  pipeline.py    # orchestration, presets
  cli.py         # typer CLI
  align/         # forced alignment — needed by eval in v1, feature in Phase 1.5
tests/
eval/
```

---

## Real-time reality (why v3 is reframed)

A "live 10x filter" on system audio is impossible: at 10x the output consumes audio ten times faster than the source produces it. Speed-up requires a source you can pull from faster than real time (a file, a buffered stream, or a player whose `playbackRate` you control). The feasible real-time component is a clarity chain running at 1:1 on already-sped audio — that is what v3 builds. The v1 offline pipeline stays the max-quality path for anything downloadable.

Real-time constraints recorded now: stateful streaming versions of every stage (overlap-add across chunk boundaries, filters carry state via `lfilter zi`), no librosa in the audio callback, no per-chunk normalization (pumping) — use a proper limiter.
