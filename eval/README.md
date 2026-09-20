# Speedman Evaluation Harness

`config.py` describes every DSP constant as a placeholder "until `eval/results.md` says
otherwise". This is what produces that file.

## Running it

Needs Media API up on `127.0.0.1:8080` (it supplies the ASR engine) and the source audio
in `/mnt/d/Output/Audio`.

```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m eval.harness
```

```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m eval.harness --clips podcast_dating --speeds 3 --presets fast,uniform
```

Results are cached in `eval/cache.json` keyed by (clip fingerprint, speed, condition,
engine), so re-runs only process what changed. `--no-cache` forces a full re-run — use it
after changing anything in `src/speedman/`, since the cache cannot see code changes.

Outputs: `eval/results.md` (human) and `eval/results.json` (machine).

## What it measures

For each (clip, speed, condition): the word error rate of ASR on the processed audio
against **ASR on the same clip at 1×**.

Using a 1× ASR pass as the reference rather than a human transcript is deliberate — it
isolates the damage compression does from the engine's own baseline error. A word
parakeet gets wrong at 1× is not a cost Speedman imposed.

The `uniform` condition is the control that matters. It is a constant-rate stretch to the
same duration: naive speed-up. Speedman's entire claim is that it beats that at the same
speed, so a preset earns its complexity only by scoring below it.

## The instrument's range — read this before trusting a number

Parakeet was not trained on time-compressed speech, so it degrades much faster than a
human listener does. Measured on `podcast_dating`, `fast` vs `uniform`:

| speed | fast | uniform | usable? |
|---|---|---|---|
| 1.5× | 0.004 | 0.004 | no — floor, nothing to measure |
| 2× | 0.014 | 0.022 | marginal |
| 2.5× | 0.057 | 0.086 | yes |
| 3× | 0.398 | 0.642 | **yes — best discrimination** |
| 3.5× | 0.817 | 0.796 | marginal |
| 4× | 0.939 | 0.896 | no — saturated |
| 5× | 0.961 | 0.971 | no — saturated |

**The harness cannot evaluate the 5–6× target band.** Above ~3.5× every condition sits
near WER 1.0, and differences are noise. `results.md` flags saturated rows and refuses to
name a winner in them.

This is a limitation of the instrument, not evidence that Speedman fails at 5–6×. What it
does establish is that the levers work where they can be measured — and it is reasonable,
though not proven, to expect the ordering to persist above the ASR ceiling.

To validate 5–6× you need one of:

- a **listening test** — the `/api/v1/compare` blind A/B endpoint already generates
  randomised sets with a hidden key, which is the right shape for this;
- an **ASR model robust to compressed speech**, e.g. fine-tuned on time-compressed audio;
- **intelligibility proxies** that do not route through a language model, e.g. STOI or
  a modulation-spectrum measure against the 1× signal.

## Sweeping a parameter

```bash
wsl --cd /mnt/d/Workspace/speedman .venv/bin/python -m eval.sweep --param silence_mult --values 1.0,2.0,2.5,3.5
```

Sweeps one `RateMapConfig` field at 3× (where discrimination peaks) and writes
`eval/sweep_results.md`. References are shared with the harness through `cache.json`.

**Read the `beats default` column, not `mean WER`.** Per-clip WER ranges from 0.03 to
0.87 within a single sweep, so a difference of means is mostly clip-difficulty noise.
Every value is scored on the same clips, so the comparison pairs per clip; the column
reports wins and a two-sided sign-test p-value. This is not a stylistic preference — in
the `silence_mult` sweep, 3.5 had the lowest mean WER (0.200 vs 0.224) but won on only
7 of 12 clips (p=0.774). The mean was one clip's outlier. The sweep will not recommend a
change unless the paired test clears p < 0.05.

## Findings so far

**Lever 1 (pause compression) is proven to work.** Setting `silence_mult = 1.0` — which
turns it off — is worse than the 2.5 default on **11 of 12 clips (p=0.006)**. No tested
value above 2.5 is significantly better, so the default stands.

**`protect_window_ms` has no measurable effect between 40 ms and 160 ms.** Nothing beats
the 120 ms default significantly, and across four runs (2.5×/3×, n=3 and n=12) the
apparent winner moved between 60, 120 and 160 while the spread of means shrank as clips
were added — the signature of an effect regressing toward zero.

That null result is worth understanding rather than dismissing. `ratemap.py`'s own
docstring predicts it: *"Raising protection widens `p` and forces everything else faster:
wider protection is weaker protection."* Because the map is normalised to hit the target
duration exactly, widening the protected region dilutes the protection proportionally.
The two effects cancel, which is why the internal decomposition changes dramatically
(onset coverage 25% → 81% across the sweep) while measured intelligibility does not.

The caveat: ASR may simply be insensitive to the consonant-transient clarity lever 2
protects. A listening test could still find a difference this metric cannot see.

## Clip set

Fixed on purpose — a moving clip set makes two runs incomparable. See `manifest.py`.
Adding a clip is safe; changing an existing clip's offset or duration invalidates every
result previously recorded against it, so prefer adding a new entry.

Sources are hours of copyrighted podcast audio and live outside the repo. A clip whose
source is missing is skipped with a note rather than failing the run.

## Why not the `eval` extra in pyproject.toml

That extra (`torch`, `torchaudio`, `transformers`, `jiwer`) pulls ~2 GB to run a wav2vec2
model, duplicating an ASR engine this workstation already has warm on port 8080.
`analyze.py` avoids torch for exactly that reason. WER is implemented directly in
`wer.py` (~30 lines, tested in `tests/test_eval_wer.py`), so the harness has no
dependencies beyond what Speedman already needs.
