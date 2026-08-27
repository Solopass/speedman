# CLAUDE.md — 10xspeedman

Ultra-speed speech engine: make sped-up speech maximally intelligible via speech-aware time compression. You are implementing **v1: the file-based Python engine**. Planning is done — read these first, in this order:

1. `docs/PLAN.md` — the roadmap; you are building Phase 1
2. `docs/ARCHITECTURE.md` — pipeline design, package layout, and the arithmetic behind the feature claims
3. `docs/TESTING.md` — the eval harness; it gates all tuning and it has a non-obvious design
4. `docs/COPILOT_CODE_REVIEW.md` — why the draft in `reference/` must not be copied
5. `docs/TRANSCRIPT_ALIGNMENT.md` — mostly later work, but `align/` is a v1 eval dependency

These docs have been through two review passes and several confident claims in earlier drafts turned out to be wrong. Where a doc marks something as unproven or unresolved, treat that as literal.

## Product target — this drives what you optimize

**Follow at ~6x, skim above.** The daily driver is 5–7x. **Concentrate tuning on N = 5–8**; do not burn effort on a 15x column nobody will use and the harness likely cannot measure. Target content is podcasts/interviews, lectures/YouTube, and audiobooks — so the eval table **reports per content type and never averages across them** (silence fraction varies ~3x between them, and lever 1's value varies with it).

**Audiobooks mean ten-hour files are normal.** Chunking is therefore a v1 requirement, not an open question to defer.

## Hard rules

- **Never copy code from `reference/copilot_v3_reference.py`.** It has syntax errors, removed librosa APIs, and concept-level DSP mistakes.
- **No pitch-shift stage, no pre-emphasis in the output path, no zero-insertion pauses, no whole-file LPC.** Each was in the draft and each is wrong (code review §§7, 9, 10, 13).
- **Analysis runs on the ORIGINAL audio.** Never detect onsets or silence on already-compressed audio.
- **One global time map, not per-segment stretch-and-join.** `ratemap.py` produces a monotonic input→output sample mapping; a backend applies it in one pass. No joins means no crossfades smearing the boundaries we are protecting, and no per-segment subprocess spawns.
- **`rubberband-cli` is a hard dependency for non-uniform mode.** It is the *only* backend with a real time-map path — librosa's `time_stretch` is scalar-only, `audiotsm` has no variable hop (and silently drops input above 2x), and `resample` under a varying rate warbles. The other three are **eval controls, not fallbacks**. Without rubberband, offer uniform stretching and fail loudly on non-uniform; do not silently approximate.
- **The boundary floor applies to boundary pauses only.** Flooring every detected gap makes lever 1 *net-harmful* in the 4–8x range (the inversion table in ARCHITECTURE.md). Classify silences by original duration; floor only those above `--boundary-ms` (~250 ms).
- **A limiter is required at the end of the post chain.** The chain as specified cannot otherwise meet its own "no sample > 0.99" criterion. Also guarantee `|y| < 1` *before* any rubberband call — pyrubberband round-trips through PCM_16 and will clip silently.
- **Nothing is assumed to work because it is plausible** — staging, transient enhancement, DRC, presence EQ, and both compression levers each ship behind a flag and earn their default from a row in `eval/results.md`. **This explicitly includes claims inherited from these docs.**
- **Read TESTING.md before building the eval harness.** Direct ASR on compressed audio does not work — CTC models have a ~50 token/s ceiling that pins every variant above ~4x, and encoder-decoder models repair and collapse. The harness measures by **re-expansion**: compress at N, restore to 1x with a fixed canonical TSM, then transcribe. Getting this wrong invalidates every tuning decision downstream.
- **All gain changes are ramped.** No hard edges anywhere in the signal path.
- Prefer proven libraries: `soundfile`, `librosa`, `audiotsm`, `pyrubberband` (+ system `rubberband-cli`), `silero-vad` or `webrtcvad`, `pyloudnorm`, `torch` + `transformers` and `jiwer` for eval, `torchaudio` for alignment.
- librosa ≥0.10: `time_stretch(y, rate=r)`, `pitch_shift(y, sr=sr, n_steps=n)` — keyword-only. `librosa.output` is gone; write with `soundfile.write`.

## What the two levers actually do — read ARCHITECTURE.md before tuning either

- **Lever 1 (pause compression)** genuinely lowers the speech rate, by `(1 − 0.6s)` where `s` is the source's silence fraction. It saturates above ~5.5x and inverts if the floor is misapplied.
- **Lever 2 (within-speech reallocation)** is **duration-neutral by construction** — normalization pins the average, so it cannot make the file shorter. It reallocates the budget from vowel centres to transients. It is scale-invariant, not unbounded: at fixed `q` it moves the wall from ~6x to ~8.6x and is spent well before 15x. `q` is a function of `N`, bounded by what the backend tolerates — and `p` and `q` trade against each other, because wider protection windows force everything else faster.
- The psychoacoustic premise under lever 2 is a **hypothesis** (the bottleneck may be central rather than peripheral). If the table says reallocation doesn't help, believe the table.

## Build order (each step: implement → tests pass → commit before next)

1. **Scaffold** — `pyproject.toml` (`speedman`), `src/speedman/` per ARCHITECTURE.md, pytest, typer CLI stub.
2. **`io.py`** — load anything via ffmpeg/soundfile → mono float32 @ 24 kHz; write wav/mp3; 16 kHz resample helper for the eval path.
3. **`stretch.py`** — `TimeStretcher` with `stretch(y, rate)` and `stretch_map(y, time_map)`; backends `resample`, `phasevocoder`, `wsola`, `rubberband`. Backends with no real time-map path **raise** rather than approximate. Staging behind `--staged`, frame/hop scaled per stage. Enforce `|y| < 1` before rubberband calls.
4. **Backend measurement — largely DONE, see `docs/SPIKE_RESULTS.md` and `eval/spikes/registration_spike.py`.** Established: exact duration control with a running residual and pinned final anchor; **10 ms anchor spacing is optimal** (finer is wasted); registration jitter floors at ~3 ms mean / ~5–9 ms max and is *not* reducible. Consequence you must build to: **protection windows are ≥80 ms of input (~150 ms preferred) — syllable-onset granularity, not phoneme.** Your job in this step is to re-run the fixture on *real speech* once `align/` exists, and with R3 (`{'-3': ''}`) before choosing an engine. The synthetic numbers are a lower bound.
5. **`eval/` + `align/`** — corpus (spanning silence fractions — LibriSpeech alone cannot measure lever 1), re-expansion harness, word-level forced alignment for the preserved-region check, `run_eval.py` → `eval/results.md` with confidence intervals. Baselines for all backends now.
6. **Calibrate the harness** (TESTING.md) — graded degradation ladder to establish resolution, monotonicity, failure-mode guards. Mark unmeasurable speed columns. Expect the top of the range to fail; that is information. **Do not tune anything before this passes.**
7. **`analyze.py` + `ratemap.py`** — `Annotation` interface (`vad` now, `aligned` later) → global time map. Boundary pauses floored, short gaps not. Transients at `q(N)` over ≥80 ms syllable-onset windows (not phoneme-width), vowel steady-state ~2x, remainder normalized to hit target duration. Handle infeasible constraint sets explicitly. Unit-test the average, the floor, the inversion guard and the infeasible case.
8. **`pipeline.py`** — apply the map in one pass; re-run eval. Should beat uniform stretching at ≥5x. **If it doesn't, that is a finding — report it, don't tune around it.**
9. **`post.py`** — presence EQ (2–5 kHz, ≤ +4 dB), optional onset-gated ramped transient enhancement, gentle DRC, EBU R128 to −16 LUFS, **limiter**. Re-run eval; keep only what beats the declared minimum effect size.
10. **Presets** (`natural`/`fast`/`aggressive`/`max`) + final table + README status. Define precedence when `--speed` and `--preset` conflict.

## Definition of done for v1

`speedman IN -o OUT --speed N [--preset P] [--backend B] [--staged/--no-staged] [--boundary-ms MS]` works on wav/mp3/m4a/mp4; suite green; `eval/results.md` shows the pipeline beating both baselines at every *measurable* speed ≥3x with CIs and unmeasurable columns marked; no clipping/NaN; duration within 2% of target; time map verified monotonic and feasible.

## Open questions to resolve during the build, not silently

- **Chunking vs. the global map — now a requirement, not an option.** Audiobooks are a primary target and ten-hour files are normal: ~3.5 GB at 24 kHz float32, and `timemap_stretch` wants the whole signal plus a temp WAV plus output. Chunk at long boundary pauses (where a seam is least damaging) and **measure the seam cost** against unchunked output on a shorter file. This must work before v1 ships.
- **Staging under a time map** is undefined (`⌈log₄(rate)⌉` assumes a scalar). If it can't be specified cleanly, ship staging for constant-rate mode only and say so.
- **Corpus size vs. decision count.** Ten short clips cannot support a dozen sequential keep/drop decisions. Declare a minimum effect size; grow the corpus before trusting close calls.

## Environment notes

- Windows user machine; develop cross-platform (pathlib, no shell-isms). ffmpeg and rubberband-cli need install — detect and report clearly.
- Python ≥3.10. `pip install -e ".[dev]"` for tests; heavy eval deps (`torch`, `transformers`, `torchaudio`, `faster-whisper`) in an `[eval]` extra. Weights download on first run — cache them; skip eval tests offline rather than failing the suite.
- Memory: a 3-hour podcast at 24 kHz mono float32 is ~1 GB. See the chunking open question.

## Style

Small pure functions on numpy arrays, type hints, docstrings that state units (samples vs seconds vs frames — most DSP bugs are unit bugs). Config via dataclasses, not module-level globals. Log stage timings at `--verbose`, and report the measured silence fraction — it predicts how much lever 1 will deliver on that file.
