# 10xspeedman — Ultra-Speed Speech Engine

Make sped-up speech intelligible. Compress time, not information: pauses get compressed harder than speech, the time budget is reallocated from redundant vowel centres to consonant transients, and formants and pitch are preserved throughout.

**The goal is not a higher speed ceiling — it is making the speed you already use stop being work.** If 5–6x is where you live and it gets muddy, that is exactly the band this targets.

---

## Quick start

**New here? Read [`GETTING_STARTED.md`](GETTING_STARTED.md)** — a step-by-step walkthrough
from install to your first listening test. On Windows, just double-click `setup.bat`.

**1. Install**

```
pip install -e ".[dev]"
```

**2. Check your setup**

```
speedman doctor
```

This verifies Python, the libraries, ffmpeg and rubberband, then runs a self-test on synthetic audio. If something is missing it tells you the exact command to install it. Start here — it is faster than debugging a failure later.

**3. Compare it against what your player already does**

```
speedman compare yourfile.mp3
```

Renders your clip eight ways — plain constant-rate stretch (roughly what a media player's speed control gives you) plus three presets, at 5x and 6x — into a folder, with **randomised filenames** and a `KEY.txt` mapping.

Blind on purpose. Knowing which file is the clever one is the fastest way to convince yourself it sounds better when it does not. Listen to `original.wav` first to reset your ear, then the rest in a random order, and rate each for **effort** rather than preference: how hard did you have to work to follow it?

**4. Once you know what you like**

```
speedman yourfile.mp3 --speed 6 --preset fast
```

**Windows:** drag an audio file onto `speedman.bat` to run the comparison without touching a terminal. Double-click it with no file to run the setup check.

---

## Install notes

**`rubberband` is required, not optional.** It is the only engine that can apply a non-uniform time map, which is the entire point of this tool. Without it you get plain constant-rate stretching — the thing we are trying to beat.

| | |
|---|---|
| Windows | download from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/), unzip, put `rubberband.exe` on your PATH |
| macOS | `brew install rubberband` |
| Linux | `apt install rubberband-cli` |

`ffmpeg` is needed for anything that is not a WAV (`winget install ffmpeg` on Windows).

---

## Commands

```
speedman FILE                          speed it up (5x, 'fast' preset)
speedman FILE --speed 6 --preset fast  pick the speed and preset
speedman FILE --uniform                plain constant-rate stretch (the control)
speedman FILE --open                   open the output folder when finished
speedman compare FILE                  render the full A/B set, blind
speedman compare FILE --speeds 4,5,6   at these speeds
speedman compare FILE --labelled       filenames say what they are
speedman doctor                        check the setup
```

Presets are `natural`, `fast`, `aggressive`, `max` — progressively harder pause compression and transient protection, and a heavier clarity chain.

Every run reports the file's **silence fraction** and the resulting **effective speech rate**. That number is the best single predictor of how well a given file will do: on a podcast that is 30% silence, a requested 6x runs the actual speech at about 4.9x, which is why it feels easier than your player's 6x. On a tightly-edited audiobook with 10% silence there is much less to win.

---

## Status

**v1 engine works; nothing is tuned yet.** The pipeline runs end to end — load → VAD annotation → global non-uniform time map → rubberband → clarity chain — with 43 tests passing. Duration accuracy is exact (0.000% error) and the limiter holds its ceiling.

Every number in `config.py` is a **placeholder**, not a tuned value. They are starting points from the reasoning in `docs/ARCHITECTURE.md`, waiting on the eval harness. If a preset sounds wrong, that is expected and useful information.

Not built yet: the eval harness (`eval/run_eval.py`), forced alignment (`align/`), and chunked processing for long files — currently capped at 90 minutes per run, with a clear message and an ffmpeg split command if you exceed it.

## Where things are

| File | What it is |
|---|---|
| `GETTING_STARTED.md` | Step-by-step: install → first listening test |
| `CLAUDE.md` | Build instructions for the next coding session |
| `docs/PLAN.md` | Roadmap v1 → v5, and the honest product target |
| `docs/ARCHITECTURE.md` | Pipeline design, and the arithmetic behind what the levers are actually worth |
| `docs/SPIKE_RESULTS.md` | Measured backend limits — what the time map can and cannot do |
| `docs/TESTING.md` | How intelligibility gets measured, and why the obvious approach fails |
| `docs/TRANSCRIPT_ALIGNMENT.md` | Using subtitles/ebooks to compress by linguistic structure |
| `docs/COPILOT_CODE_REVIEW.md` | Audit of the original draft — what was broken and why |
| `src/speedman/` | The engine |
| `eval/spikes/` | Measurement fixtures — re-run on any backend change |
| `reference/` | The original Copilot draft, kept for reference only — **do not run or copy from it** |

## How it works

Past ~2.5x, naive speed-up becomes mush — not because information is gone, but because the cues the brain uses to segment speech get smeared. Two levers fix different parts of that:

**Pause compression** genuinely lowers the rate the speech experiences: with silence fraction `s`, speech runs at about `N × (1 − 0.6s)`. Real, bounded, and it saturates above ~5.5x.

**Within-speech reallocation** cannot make the file shorter — the map is normalised to hit the requested length exactly — but it moves the time budget from redundant vowel centres to consonant transients, so the consonants inside a 6x file live in a slower regime than 6x.

Honest headline: together these buy roughly **1.5–3x of effective headroom** over naive speed-up, not an order of magnitude. That is enough to make 6x feel like 4.5x, which is the entire point. See `docs/ARCHITECTURE.md` for the arithmetic, including where each lever stops paying.

---

## License

**Source-available, noncommercial.** Copyright © 2026 Solopass. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

- ✅ **Free** for personal use, hobby projects, study and research, and for nonprofits, schools and public institutions.
- 💼 **Commercial use** (in a business, product or paid service, or for-profit internal use) needs a paid license. See [COMMERCIAL.md](COMMERCIAL.md), or contact [realsolopass@gmail.com](mailto:realsolopass@gmail.com) · <https://polymatica.pages.dev>.
