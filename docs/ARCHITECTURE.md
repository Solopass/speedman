# Speedman Architecture & Arithmetic

## The Problem with Naive Speed-up

Past ~2.5× speed, naive time-stretching turns human speech into unintelligible mud. This is not because information is missing, but because:
1. **Consonant transients smear**: Plosives (/p/, /t/, /k/) and fricatives (/s/, /sh/) become too brief for auditory temporal integration.
2. **Pauses waste proportional time**: A 300ms pause in a 1× recording still consumes 60ms at 5×, stealing valuable time budget from the words.

## The Two Levers

Speedman operates through two distinct mathematical levers:

### Lever 1: Pause Compression
Pauses between sentences and phrases are compressed significantly harder than speech. As a
first approximation, for a file with silence fraction $s$:
$$\text{Effective Speech Rate} \approx N \times (1 - 0.6s)$$

**This estimate is now superseded by measurement.** `pipeline.process` reports
`effective_speech_rate` read directly off the solved time map
(`ratemap.TimeMap.measured_speech_rate`), because the formula above ignores the preset and
runs optimistic — by ~4% at 5×, and by 11% at 20×. The estimate is still reported as
`estimated_speech_rate` so the two can be compared.

This section previously assumed a typical podcast at $s = 0.30$. **Measured across all 12
evaluation clips, real content is $s = 0.133$ to $0.153$** — roughly half (see
`docs/EVALUATION.md`). Two podcasts and an audiobook all landed in that band, so the lower
figure is the one to reason from.

Working the arithmetic at the measured $s \approx 0.14$:
- A requested $N = 6.0\times$ runs speech at about $5.5\times$, not $4.9\times$.
- Lever 1 therefore buys roughly $1.1\times$ of headroom, not $1.1$–$1.5\times$.

That is a real effect and a measured one — disabling pause compression is worse on 11 of 12
clips ($p = 0.006$) — but it is about half what this document originally claimed.

### Lever 2: Within-Speech Time Reallocation
The global duration is normalized to match the requested speed multiplier exactly. Therefore, Lever 2 does not shorten the file—it reallocates duration within speech:
- Consonant transients and syllable onsets are protected and given extra time.
- Redundant steady-state vowel centers are compressed harder.

## Why Rubberband?

Rubberband is the only open-source audio engine capable of non-uniform time-mapping without altering pitch or introducing phase vocoder smearing.
Speedman calculates an array of $(T_{\text{in}}, T_{\text{out}})$ anchors and passes them directly to Rubberband's map mode.

## System Design on Workstation

```
[HTTP Request / Desktop Launcher]
              │
              ▼
   Systemd Socket (127.0.0.1:8081)
              │
              ▼ (On-demand activation)
   Speedman FastAPI Service
    - Pinned to E-Cores 8–15
    - nice -n 10, ionice -c 3
    - Auto-close watchdog (60s idle)
              │
      ┌───────┴───────┐
      ▼               ▼
Single File      Blind A/B Set
(D:\Audio\Speed) (D:\Audio\Speed\comparisons)
```
