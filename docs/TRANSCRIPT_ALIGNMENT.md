# Transcript-assisted compression

## The idea

When the words are known — subtitles on a video, an SRT/VTT file, an ebook alongside its audiobook — the pipeline can stop inferring structure from the waveform and start reading it from the text.

Everything in v1's time map rests on `analyze.py` guessing: VAD says "this is silence", onset strength says "something started here". Both are heuristics with meaningful error rates. A transcript plus forced alignment replaces them with word boundaries and, with the right aligner, phoneme boundaries — turning the annotation from a heuristic guess into a **measured estimate with a much lower error rate**.

Not "ground truth". Forced alignment is a model output and it is wrong sometimes; see Failure modes.

**The hypothesis this rests on** — that v1's levers are limited by annotation quality rather than by the stretching itself — is *plausible but untested*. If v1's eval shows the pipeline losing more to backend artifacts than to misplaced protection windows, this phase is lower value than it looks. That comparison is cheap to make once v1 exists, and it should be made before this phase is scheduled.

## Important: subtitles are a prior, not the annotation

Raw subtitle timings cannot drive a time map. SRT/VTT cues are per-caption-line, authored for readability, and routinely off by ±0.5 s — an eternity when a protected consonant burst is 60 ms. Using cue timings directly would misplace every protection window.

The subtitle's value is its **text**. Forced alignment — text plus audio in, precise boundaries out — does the timing work; cue timings serve only as a cheap prior that makes alignment fast and robust on long files. YouTube auto-captions carry genuine per-word timings but are ASR output and inherit ASR errors, so re-align them too.

---

## What alignment realistically buys, and at what cost

This splits into two tiers with very different price tags. Earlier drafts of this document conflated them and promised the expensive tier at the cheap tier's cost.

### Tier 1 — word-level alignment (cheap, available now)

`torchaudio.functional.forced_align` is CTC forced alignment using the same wav2vec2 model the eval harness already loads. No new heavy dependency. It gives **word and character boundaries at a 20 ms quantum** (wav2vec2-base's frame stride).

Note the honesty problem that quantum creates: this document rejects subtitle cues for being imprecise relative to a 60 ms event, and 20 ms is still ±33% of that same event. **Tier 1 is precise enough for word-scale decisions and not precise enough for phoneme-scale ones.** What it supports:

- **Surprisal-weighted compression.** With text in hand, per-word information content is computable — run a small LM (`distilgpt2` suffices) and take each token's surprisal. High-surprisal words (proper nouns, numbers, topic-bearing content) get protected; low-surprisal ones get crushed. This attacks "compress time, not information" at the **semantic** level, which nothing else in the roadmap does.
- **Prosodic pause allocation.** Sentence and clause boundaries come from punctuation rather than from pause length, so boundary pauses are identified by *syntax* instead of by the duration threshold v1 uses. This also sidesteps much of the stop-closure trap: a closure is never at a punctuation mark.
- **Content reduction — the actual mechanism above 6x.** The product target is follow-at-6x, skim above, and above 6x the binding constraint stops being audio quality and becomes linguistic units per second. The fix is fewer words, not faster ones. Cheapest tier is near-lossless: strip filler, false starts and repetitions (6–10% of words in spontaneous speech) and delete long pauses outright instead of flooring them — call it 1.2–1.3x free on unscripted audio, which stacks multiplicatively with the DSP to put ~7x effective within reach while the *audio* stays at a followable 5–6x. Worthless on audiobooks, strong on podcasts and lectures. Beyond that tier, dropping low-surprisal spans trades information for speed and belongs behind an explicit flag.
- **Content-aware skimming** (`--mode skim`) for 8x+, built on the above. Keep sentence-initial and high-surprisal words near a followable rate; crush filler hard. This is arguably a distinct product mode, not a preset, and it is the most likely place for the top of the speed range to become genuinely useful.

Two honest caveats on surprisal weighting, both for the eval table to settle:

- **Less headroom than it looks** — natural speech already shortens predictable words, so some of this reallocation has been done by the speaker.
- **It may backfire** — function words are low-information individually but carry syntactic structure, and listeners use them to parse clause boundaries. Crushing them could hurt comprehension while improving any word-level metric. Watch the gap between WER and the listening rubric here specifically.

### Tier 2 — phoneme-level alignment: DROPPED (see docs/SPIKE_RESULTS.md)

**Do not build this.** Measurement, not argument: rubberband places time-map windows with ~5 ms jitter and they must be ≥80 ms wide to be robust, so protection works at syllable-onset granularity. Phoneme boundaries at 10 ms precision are **precision the mechanism cannot consume.** Word-level alignment from `torchaudio.functional.forced_align` (20 ms quantum, no heavy dependency) already exceeds what is usable. Dropping this removes the Montreal Forced Aligner dependency entirely.

Revisit only if the stretching engine's registration floor improves by an order of magnitude — that would be a different engine, not a different aligner.

The original reasoning is kept below because the phoneme-class model still explains *why* the rate map is shaped the way it is, even though it is now applied at coarser granularity.

#### Original Tier 2 rationale (retained for context)

The phoneme-class rate table below is the version of lever 2 with real linguistic grounding:

| Class | Treatment | Why |
|---|---|---|
| Boundary pause / true silence | crush hardest | little linguistic information (lever 1) |
| Stop closure | **protect** | phonemically part of the consonant, not a pause |
| Plosive burst, affricate onset | protect most | high information density |
| Fricatives | protect moderately | spectrally distinctive but comparatively long |
| Diphthong / formant transitions | protect | the trajectory *is* the vowel identity |
| Nasals, liquids, glides | compress moderately | redundant given context |
| Vowel steady-state centre | crush | maximally redundant (lever 2's donor region) |

**This table cannot be built from Tier 1, and an earlier draft of this document claimed otherwise.** The claimed route — map character boundaries through CMUdict or `g2p-en` — does not work:

- CMUdict and g2p give a word's phoneme **sequence**, not the **times between phonemes**. `SPIN → S P IH N` carries no timing.
- Character boundaries do not map onto phoneme boundaries. English orthography is not phonemic: "though" is 6 characters and 2 phonemes, "x" is 1 character and 2 phonemes, silent "e" is 1 character and 0 phonemes. There is no correspondence to transfer times across.
- Resolving a 50–80 ms closure *from* its following burst needs ~10 ms precision. The 20 ms quantum cannot.

**Tier 2 therefore requires the Montreal Forced Aligner or a phoneme-level CTC model** — a large external install with its own acoustic models and pronunciation dictionaries. That is a real cost, and it must be weighed on its own rather than smuggled in as a free consequence of Tier 1.

### On the stop-closure argument

The stop-closure case is still the clearest illustration of why annotation quality bounds this whole design — VAD reads a phonemically critical near-silence as a compressible pause, so lever 1 crushes exactly what lever 2 is protecting. But two corrections to how earlier drafts stated it:

- **The "spin" example was wrong.** In `/sp/` English *neutralizes* the voicing contrast — there is no /b/–/p/ opposition after /s/, and the /p/ is unaspirated precisely because of that. It is the one environment where the claim cannot hold. Use a word-medial contrast instead (*rapid* vs *rabid*), where closure duration and voicing during closure genuinely distinguish the stops.
- **Closure duration is a secondary cue.** VOT, voicing during closure (the voice bar), and preceding-vowel duration are the primary ones. Damaging closure duration degrades the cue set; it does not destroy the contrast single-handedly.

And note that v1 already mitigates most of this for free: the boundary-duration threshold (~250 ms, ARCHITECTURE.md) excludes typical 50–80 ms closures from flooring by construction. Tier 2 makes it exact; it is not rescuing a broken v1.

---

## Alignment as eval infrastructure (this comes first)

The **preserved-region check** in TESTING.md — the only direct measurement of whether lever 2 reallocated the time budget as intended — requires forced alignment on the original audio to compute at all.

So `align/` is needed as a **v1 eval tool** regardless of whether the user-facing feature ever ships, and CLAUDE.md's build order includes it on that basis. Building it there front-loads the value and means the feature later arrives already exercised on real audio.

## Mechanism

```
transcript source (SRT / VTT / ebook text / ASR fallback)
  → normalize text (strip cue markup, speaker labels, sound descriptions)
  → forced alignment against the ORIGINAL audio
  → word spans (Tier 1) or phoneme spans (Tier 2) with confidence
  → optional: per-word surprisal from a small LM
  → Annotation  (same structure the vad source emits)
  → ratemap.py  (unchanged — this is why the interface exists)
```

## Failure modes

Alignment is not free and not always right. It must degrade to the `vad` source rather than produce a bad time map:

- **Transcript/audio mismatch** — abridged audiobooks, subtitles for a different cut, missing chapters. Detect via alignment confidence and per-region drift; fall back on the affected span, not the whole file.
- **Long files** — a 10-hour audiobook must be chunked with overlap and stitched. `torchaudio.functional.forced_align` is batch-size-1 only.
- **Non-speech regions** — music, applause, effects have no transcript. Detect and hand back to VAD.
- **Multi-speaker overlap** — alignment degrades on crosstalk; treat low-confidence spans as unannotated.
- **Wrong language / heavy accent** — confidence check, then fall back.

Rule: **a low-confidence alignment must never silently produce a worse time map than no alignment at all.** The fallback threshold is a tuned parameter with a row in the eval table.

## The obvious objection

*If you already have the text, why listen at all — just read it.*

Because the use case is eyes-free and hands-free: commuting, walking, cooking, working out. The transcript is not a substitute for the audio; it is metadata that makes the audio compressible with less damage.

A related option deliberately **rejected**: transcribe and re-synthesize with fast TTS. That is how screen-reader users actually exceed 6x, and it would beat TSM at those rates — but it discards the speaker's voice and prosody, which for podcasts and interviews is much of the point. Out of scope given a follow-at-6x target. Revisit only if the goal ever changes to maximum rate at any cost.

## Roadmap position

**`align/` (word-level, for eval): v1.** **Tier 1 features: Phase 1.5.** **Tier 2: dropped — the mechanism cannot use phoneme-level precision (SPIKE_RESULTS.md).**

The `Annotation` interface all of this plugs into is defined in v1 — anticipating it is an afternoon, retrofitting it is a rewrite.

Order:

1. `align/` module + word-level forced alignment on original audio, exposed to `eval/run_eval.py` for the preserved-region check. *(v1)*
2. SRT/VTT/text ingest and normalization; audio↔transcript matching and confidence.
3. `aligned` annotation source at word level: syntax-driven boundary pauses. **Eval delta vs. `vad` — this is the proof, and it decides whether Tier 2 is worth funding.**
4. Surprisal weighting, behind a flag, measured separately (it may lose).
5. `--mode skim` for 10x+.
6. ~~Evaluate MFA for Tier 2 phoneme classes~~ — dropped; see SPIKE_RESULTS.md.
