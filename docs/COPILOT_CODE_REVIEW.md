# Copilot Draft Code Review

The original draft lives at `reference/copilot_v3_reference.py`. It is a useful idea sketch and a poor code source: it would not run as written, and several stages are conceptually wrong. **Do not copy code from it.** This review exists so the implementation keeps the good ideas and avoids re-importing the bugs.

## Verdict summary

Keep the ideas: modular pipeline, protect formants/transients/pauses, hybrid TSM concept, speed presets, phased roadmap. Discard the implementations: every stage needs a rewrite, and three stages (EQ, LPC, pitch) are wrong at the concept level, not just the code level.

## Hard errors (would crash or was removed from the API)

1. `librosa.output.write_wav(...)` — removed from librosa in 0.8 (2019). Use `soundfile.write`.
2. `chunk.reshape(1, -)` — Python syntax error (appears twice; should be `(1, -1)`).
3. `librosa.effects.time_stretch(y, rate)` / `pitch_shift(y, sr, n_steps)` / `phase_vocoder(D, rate)` — these are keyword-only args in current librosa (`rate=`, `sr=`, `n_steps=`). Positional calls raise TypeError.
4. Hybrid blend `mask * y_wsola + (1 - mask) * y_pv` — `mask` has the original signal's length; `y_wsola`/`y_pv` are ~len/rate. Shape mismatch → crash. (Concept problem too — see §8.)
5. Real-time callback writes `process_chunk(y)` into `outdata[:, 0]` — after 10x compression the result is ~1/10 of `frames` samples. Shape mismatch → crash every callback.

## Concept-level errors (would run after fixes, but do the wrong thing)

6. **The "adaptive EQ" destroys the signal.** `sig.iirpeak` designs a narrow band-PASS filter; the code replaces the signal with only its 4.5 kHz band, then scales it. A clarity EQ must be additive: `y + gain · bandpass(y)` (or a proper peaking biquad). As drafted, this stage outputs ringing filtered noise.
7. **Whole-file LPC preserves nothing.** One LPC fit over an entire recording yields an average filter; formants change every ~20 ms with each phoneme. Formant preservation requires frame-based analysis/synthesis with overlap-add — or better, skip hand-rolled LPC entirely and use pyworld (v2) or rubberband `--formant` (v1). The hand-rolled Levinson–Durbin also has a sign-convention bug (`residual = lfilter(a, [1], y)` with positive `a[i]` is not the prediction-error filter), so the synthesis filter can be unstable. If LPC is ever needed, `librosa.lpc` exists.
8. **Blending two independently stretched signals doesn't hybridize them.** WSOLA and PV outputs are not sample-aligned (WSOLA shifts segments to maximize correlation), so a sample-wise crossfade of the two full outputs produces comb-filtered garbage. Real hybrid TSM picks the algorithm per time segment during synthesis and crossfades at segment joins.
9. **The pitch stage solves a nonexistent problem.** TSM (WSOLA/PV) does not shift pitch — avoiding the chipmunk effect is why TSM exists. The unconditional-ish −3 semitone "normalization" would deepen every voice. Also, `pyin` mean-pitch of 250 Hz simply means a typical female speaker, not an artifact. Delete the stage.
10. **Pre-emphasis with no de-emphasis** — +6 dB/octave tilt baked into the output; everything sounds thin and harsh. Analysis-only trick, wrongly placed in the output path.
11. **"Spectral envelope protection" is a blur filter.** Median-filtering the dB spectrogram and resynthesizing REPLACES the magnitude with a smoothed copy, deleting harmonic fine structure. It doesn't protect the envelope; it smears everything toward it.
12. **Analysis runs on the wrong signal.** Onset detection, VAD-ish energy gating, and pause detection all run on already-compressed audio, where onsets are smeared and silences are 10x shorter than the detector's frame. Analyze the original, apply to the stretch plan (see ARCHITECTURE.md).
13. **Zero-insertion pauses click.** Hard-edged zero splices at 30 ms boundaries produce audible clicks; also the energy threshold basically never fires post-compression. Superseded by rate-map silence floors.
14. **Transient boost has no ramps** — a flat ×1.4 over 12 ms windows clicks at both edges. Gains need attack/release ramps.
15. **Real-time plan is infeasible as specified.** Beyond §5: librosa `pyin`/onset/STFT inside a 40 ms callback exceeds the time budget by orders of magnitude; `time_stretch` with default n_fft=2048 can't even process a 40 ms (960-sample) chunk; per-chunk peak normalization pumps. And conceptually, a live passthrough cannot speed audio up at all (10x consumes input faster than it arrives). Correct framing in ARCHITECTURE.md §Real-time reality.

## What the draft got right (keep these)

Modular stage decomposition; the preservation checklist (formants, transients, prosody, pitch, envelope); WSOLA-for-transients / PV-for-steady-state intuition; onset-strength as the transient signal; speed presets; ONNX enhancement as a later phase; the phased v1→v5 shape of the roadmap.

## What the draft missed entirely (now core to the plan)

Non-uniform time compression (compress silence harder than speech) — the highest-leverage idea in this whole problem space; staged stretching for extreme rates; objective evaluation (ASR WER) so tuning doesn't depend on ears; existing high-quality TSM libraries (rubberband, audiotsm, pyworld) instead of hand-rolling everything.

---

## Note added after later review (§8 and the no-crossfade rule)

§8 above prescribes hybrid TSM that *"picks the algorithm per time segment during synthesis and crossfades at segment joins"*. That conflicts with the hard rule in CLAUDE.md and ARCHITECTURE.md that there are no segment joins and no crossfades in the signal path — a 10 ms crossfade at 10x spans most of a compressed phoneme.

The conflict is real and is **not** dissolved by the global time map: switching backends mid-signal produces different waveform and phase continuations at the switch point, so a per-region backend switch *is* a join and needs a crossfade.

Resolution: **the no-crossfade rule wins for v1.** Hybrid TSM is therefore out of scope until someone specifies how to switch algorithms without a join (phase-continuation handoff, or synthesising both and switching only at a zero-crossing in a low-information region). Phase 2 must treat this as an open design problem rather than a scheduled feature.
