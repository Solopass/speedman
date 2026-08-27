# Getting started — your first listening test

Everything here is Windows-first, since that is what you are on.

---

## Step 1 — Set it up (once, ~5 minutes)

**Double-click `setup.bat`.**

It installs Python packages into a private environment inside the project folder (nothing touches your system Python), offers to download the Rubber Band audio engine, and then runs a self-check.

You want it to finish with:

```
  OK   Python         3.12.x
  OK   numpy          installed
  ...
  OK   rubberband     ...\10xspeedman\bin\rubberband.exe

running a self-test on synthetic audio ...

Everything works. (5x test: length 0.00% off, peak 0.97)
```

If anything says `FAIL`, it prints the exact fix underneath. The two that actually happen:

**"Python was not found"** — install Python 3.10+ from [python.org/downloads](https://python.org/downloads) and **tick "Add python.exe to PATH"** in the installer. That checkbox is off by default and is the single most common setup failure. Then run `setup.bat` again.

**"rubberband MISSING"** — the auto-download did not work (corporate network, firewall, changed URL). Do it by hand:
1. Open [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/)
2. Download the **command-line utility** zip
3. Open the zip, find `rubberband.exe` inside
4. Copy it into the `bin\` folder in this project
5. Double-click `setup.bat` again

You never need to edit your PATH. The `bin\` folder is checked automatically.

**ffmpeg** is only needed for mp3/m4a/mp4. If it says MISSING and you only have WAVs, ignore it. Otherwise: `winget install ffmpeg` in a terminal.

---

## Step 2 — Pick a clip

**30–60 seconds** is the sweet spot. Long enough to settle into, short enough to listen to eight times without going mad.

Choose something **you already know gets muddy at 5–6x**. That is the actual test. A clean studio-read audiobook is the hard case for this tool and a bad first impression; a two-person podcast with natural pauses is where it should shine.

Put it anywhere. Any format ffmpeg reads.

If you have a longer file, cut a chunk out first:

```
ffmpeg -i podcast.mp3 -ss 00:10:00 -t 45 clip.mp3
```

(Starts at 10 minutes, takes 45 seconds. Pick a section that is mostly talking, not intro music.)

---

## Step 3 — Build the comparison

**Drag your audio file onto `speedman.bat`.**

That renders eight versions plus the original into a folder called `<yourfile>_compare`, and opens it.

Or from a terminal in the project folder:

```
.venv\Scripts\activate
speedman compare clip.mp3
```

Useful variations:

```
speedman compare clip.mp3 --speeds 4,5,6,7    more speeds
speedman compare clip.mp3 --labelled          filenames say what they are
speedman clip.mp3 --speed 6                   just one file, no comparison
```

---

## Step 4 — Listen (this part has rules)

The filenames are deliberately meaningless — `a0.wav`, `b1.wav` and so on. `KEY.txt` in that folder says which is which.

**Do not open `KEY.txt` yet.** Knowing which file is the clever one is the fastest way to convince yourself it sounds better when it does not, and your ears are the only instrument we have at this stage.

1. Play `original.wav` first to reset your ear.
2. Play the eight files **in a random order** — not alphabetical, since alphabetical groups them by speed.
3. For each one, note a number **1–5 for effort**: how hard did you have to work to follow it? Not whether you liked it — effort. A file can sound slightly odd and still be easy to follow, and that is a win.
4. Also note anything specific you hear: harsh or sibilant, warbly or metallic, words running together, clicks, pauses feeling wrong.

Something like:

```
a0  3   a bit harsh on s sounds
a1  2   easiest so far
b0  4   words blur together
...
```

**Then** open `KEY.txt` and see what you picked.

---

## Step 5 — Tell me what happened

What I need, roughly in order of usefulness:

1. **Your ratings before you opened the key.** Even just "a1 and c1 were easiest, b0 was worst."
2. **How the presets ranked against `uniform`.** `uniform` is the plain constant-rate stretch — roughly what your podcast app's speed control does. That is the one to beat. If it wins, that is a real and important result and I want to know immediately.
3. **The specific artifacts** you heard, and on which files.
4. **The line `speedman` printed** about silence fraction, e.g. `32% of the original is silence, so the speech itself runs at about 4.85x`.

Every number in `src/speedman/config.py` is a placeholder right now, reasoned rather than tuned. Your ratings are the first real evidence any of them have ever had. "The aggressive one was unlistenable" is genuinely useful — it tells me which parameter went too far.

---

## If something breaks

**Run `speedman doctor` first.** It catches most things and tells you the fix.

| What you see | What it means |
|---|---|
| `'speedman' is not recognized` | The environment is not active. Run `.venv\Scripts\activate` first, or just use `speedman.bat`. |
| `rubberband ... was not found` | Put `rubberband.exe` in `bin\`. See step 1. |
| `needs ffmpeg` | Your file is not a WAV. `winget install ffmpeg`, or convert to WAV first. |
| `this file is N minutes long` | Over the 90-minute cap. Chunked processing is not built yet; cut a section out with the ffmpeg command in step 2. |
| It seems to hang on "analysing" | Normal on a long file — roughly 20 seconds per hour of audio. Under a minute, something is wrong. |

For a full traceback on any error, add `--debug`.

---

## What this is and is not, yet

The engine works and is tested, but **nothing is tuned**. You are not evaluating a finished product; you are producing the first data that tunes it.

Realistic expectation: this should make 6x feel closer to 4.5x. It will not make 10x followable — nothing will, that limit is in your head rather than in the audio. If 6x feels meaningfully easier than your podcast app's 6x, it is working.
