# Master Plan — v1 → v5 (revised)

This is the Copilot roadmap, kept in spirit but corrected where it was technically wrong, then revised again after a second review.

Changes from the original Copilot plan: (1) non-uniform time compression is the centerpiece of v1 — it was missing entirely and it is the largest intelligibility win; (2) the "real-time live system audio at 10x" concept is reframed, because it is physically impossible as stated (ARCHITECTURE.md §Real-time reality); (3) goals are stated honestly against the human comprehension ceiling.

Changes from the second review: (4) the two non-uniform levers are separated, and the arithmetic showing what each is actually worth is now in ARCHITECTURE.md — pause compression saturates above ~5.5x and *inverts* if its floor is misapplied, and within-speech reallocation is duration-neutral by construction and scale-invariant rather than unbounded — together they buy perhaps 1.5–3x of effective headroom, not 10x; (5) segment-wise stretch-and-join is replaced by a single global time map (correctness *and* a serious performance trap avoided); (6) staged stretching becomes a measured flag rather than an assumption; (7) the eval harness measures by re-expansion rather than by transcribing compressed audio directly — CTC models have a ~50 token/s ceiling that pins every variant above ~4x, and encoder-decoder models repair and collapse — and it must be calibrated against a graded degradation ladder before anything is tuned against it; (8) new Phase 1.5 — transcript-assisted compression, with `align/` pulled into v1 as eval infrastructure; (9) `rubberband-cli` is now a hard dependency for non-uniform mode, because it is the only backend with a real time-map path.

## Product target (decided 2026-08-27 — this drives every tuning priority)

**Follow at ~6x. Skim above.** The daily driver is 5–7x, where the listener genuinely takes in the content. 8x+ is a skim mode for catching topic shifts, names and structure — not for following. 10–15x is benchmarked, never promised.

This is deliberate, and it matches where the physics actually pays. The DSP levers are worth the most in the 4–8x band and are largely spent by 10x (ARCHITECTURE.md). **So tuning effort concentrates on N = 5–8.** Do not spend days optimizing a 15x column that no one will use and that the eval harness probably cannot even measure.

Above ~6x the useful lever stops being signal processing and becomes **content reduction** — fewer words rather than faster ones. That is the mechanism for skim mode, and it lives in Phase 1.5 with the transcript work.

### Target content

Podcasts and interviews, lectures / talks / YouTube, and audiobooks. Three consequences that are not obvious:

- **Lever 1's value varies ~3x across these.** Podcasts and interviews carry `s ≈ 0.20–0.35` silence; lectures ~0.20–0.30; audiobooks ~0.10. The eval table therefore **reports per content type and never averages across them** — an averaged number would hide that the feature does most of its work on two of the three.
- **Transcripts are available for all three** (podcast transcripts, YouTube captions, the ebook beside the audiobook). That makes Phase 1.5 unusually high-value here rather than a nice-to-have.
- **Audiobooks are the long-file case.** Ten-hour files are the normal case, not the edge case. This promotes the chunking question from "open" to "must solve before v1 ships" — see ARCHITECTURE.md.

### Quality bar per speed

- **2x–3x** — indistinguishable from someone who just talks fast. No artifacts.
- **4x–6x** — fully intelligible with attention. **This is the product.**
- **8x–10x** — consonants and boundaries preserved; skimming and training headroom.
- **12x–15x** — as clean as the physics allow; benchmarked, not promised.

### The ceiling, stated honestly

Trained listeners of time-compressed speech top out around 3x–6x even with perfect audio, and the classical literature puts comprehension breakdown near 275–300 wpm, attributed to central processing rate rather than audio quality. Screen-reader users who exceed that are listening to *synthesizers generating fast speech directly*, not to compressed recordings — a different technique with different tradeoffs, out of scope here.

So above ~6x, better DSP stops being the binding constraint. Anyone tempted to chase 15x with signal processing should read this paragraph again.

## Phase 1 — v1: File-based engine (BUILD THIS FIRST)

Goal: `speedman input -o output --speed N` producing best-in-class results at 2x–10x. Detailed build order lives in `CLAUDE.md`; this is the shape of it.

1. **Scaffold** — `pyproject.toml` (`speedman`), `src/speedman/` per ARCHITECTURE.md, pytest, typer CLI stub.
2. **I/O** — load anything (wav/mp3/m4a/mp4 via ffmpeg), mono float32 @ 24 kHz; write wav/mp3; 16 kHz helper for the eval path.
3. **Time-stretch backends behind one interface** — `resample` (control), librosa phase vocoder (baseline), WSOLA via `audiotsm`, and Rubber Band via `pyrubberband`. **Only rubberband has a real time-map path** (`timemap_stretch`, verified working); librosa's `time_stretch` is scalar-only and `audiotsm` has no variable hop and silently drops input above 2x. The other three are **eval controls, not fallbacks** — backends without a time-map path raise rather than approximate. `--formant` is inert without pitch shifting and is not used. Staging behind `--staged`, frame scaled per stage.
4. **Measure the backend before building on it** — anchor precision (running residual, pinned final anchor) and time-map registration error. A preliminary test showed rubberband honouring key frames only to ~6.6 ms; at 10x that is a whole phoneme of input. If protection windows cannot be placed precisely, the lever-2 design must change — find out now, not in step 8.
5. **Eval harness + `align/`** — corpus spanning silence fractions (LibriSpeech alone *cannot* measure lever 1 — it has almost no silence), the re-expansion protocol, word-level forced alignment for the preserved-region check, `run_eval.py` → `eval/results.md` with confidence intervals. Baselines for all backends now.
6. **Calibrate the harness** (TESTING.md) — graded degradation ladder to establish resolution, monotonicity, failure-mode guards; mark unmeasurable speed columns. **Nothing is tuned before this passes.**
7. **Annotation + rate map (the key feature)** — `Annotation` interface with the `vad` source; `ratemap.py` builds one global monotonic time map. Boundary pauses (> ~250 ms) compressed ~2.5x harder and floored; **short inter-word gaps are not floored** — flooring everything makes lever 1 net-harmful in the 4–8x range. Transients protected at `q(N)`, vowel steady-state crushed, remainder normalized to the requested duration. Handle infeasible constraint sets explicitly.
8. **Apply the time map** in a single pass; re-run eval. Should beat uniform stretching at ≥5x — and if it doesn't, that is a finding, not a bug to tune away.
9. **Post chain** — presence EQ (2–5 kHz, ≤ +4 dB), optional onset-gated ramped transient enhancement, gentle DRC, EBU R128 to −16 LUFS, and a **limiter** (without which the chain cannot meet its own no-clipping criterion). Keep only what beats the declared minimum effect size.
10. **Presets** + final eval table + README status; define `--speed`/`--preset` precedence.

Done when: the table shows the pipeline beating naive resample and plain librosa at every *measurable* speed ≥3x, with CIs, and 5x output is comfortably intelligible by ear.

**Three things are deliberately left open** and must be resolved during the build rather than assumed away: chunking a 3-hour file against a whole-signal time map; how staging is even defined under a time map; and whether a ten-clip corpus can support a dozen sequential keep/drop decisions. See CLAUDE.md §Open questions.

## Phase 1.5 — Transcript-assisted compression

Goal: stop guessing where the information is.

When a transcript exists (subtitles, VTT/SRT, an ebook beside its audiobook), forced alignment gives word and phoneme boundaries, and the rate map is assigned by linguistic class instead of by onset energy. This fixes failures no DSP tuning can — most starkly the stop-closure case, where VAD reads a phonemically critical silence as a compressible pause. Also unlocks surprisal-weighted compression, prosodic pause allocation, and a content-aware `--mode skim` for 10x+.

Full design, caveats and failure modes: **`docs/TRANSCRIPT_ALIGNMENT.md`**.

Placed after v1 because its value can only be shown as a delta over the VAD baseline on a validated table. But the `Annotation` interface it plugs into is defined in v1 — anticipating it is an afternoon, retrofitting it is a rewrite. The alignment module itself is needed even earlier, as eval infrastructure for the preserved-region check.

## Phase 2 — v2: Formant-true resynthesis

Goal: make 8x–12x stop sounding phasy/metallic.

- **WORLD vocoder path** (`pyworld`): decompose to f0 + spectral envelope + aperiodicity, decimate frames in time, resynthesize. Formants preserved by construction. Replaces the Copilot plan's broken whole-file-LPC idea with the tool built for the job. Enters as another backend behind the existing interface.
- **Hybrid TSM** — switch algorithm by region (WSOLA on transients, PV/WORLD on voiced steady state). Not the Copilot version, which blended two unaligned full-length outputs (code review §8). With a global time map this is a per-region backend selection, not a re-join.
- Prosody-aware pause model — largely delivered by Phase 1.5 if a transcript is available; this is the no-transcript version.

## Phase 3 — v3: Real-time clarity engine + player integration

Goal: use it on YouTube/Netflix/podcasts, live.

Reframed: the *speed change* happens at the media player (`playbackRate` — browsers pitch-correct already); our DSP runs in real time at 1:1 on the already-sped audio as a *clarity* chain (EQ, transient enhancement, envelope repair). That is feasible; a 10x passthrough of live audio is not.

- Streaming chunk architecture with overlap-add state (sounddevice), <50 ms latency.
- Browser extension prototype: `video.playbackRate` + WebAudio worklet clarity chain. Note that a browser extension can read the page's subtitle track — Phase 1.5's annotation source is available here.
- For seekable/downloaded media, "pre-roll" mode: process ahead of the playhead with the full offline pipeline for max quality.

## Phase 4 — v4: ML-enhanced clarity

- Off-the-shelf ONNX speech enhancement (DeepFilterNet-class) post-stretch; measure WER impact before inventing anything.
- If warranted: a small model trained on (clean-fast, artifacted-fast) pairs generated synthetically from any speech corpus — unlimited training data by construction.

## Phase 5 — v5: Product

- Browser extension (WebAudio + WASM port of the v3 chain) — likely first, since it meets users where the content is, and where subtitle tracks already exist.
- Desktop app (Electron or Tauri + Rust DSP core) and/or mobile later.

## Immediate next step

Build Phase 1, steps 1–9, in order. Each step lands with tests and an eval-table update before the next. See `CLAUDE.md`.
