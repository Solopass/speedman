# Speedman Architecture & Arithmetic

## The Problem with Naive Speed-up

Past ~2.5× speed, naive time-stretching turns human speech into unintelligible mud. This is not because information is missing, but because:
1. **Consonant transients smear**: Plosives (/p/, /t/, /k/) and fricatives (/s/, /sh/) become too brief for auditory temporal integration.
2. **Pauses waste proportional time**: A 300ms pause in a 1× recording still consumes 60ms at 5×, stealing valuable time budget from the words.

## The Two Levers

Speedman operates through two distinct mathematical levers:

### Lever 1: Pause Compression
Pauses between sentences and phrases are compressed significantly harder than speech. If a file has silence fraction $s$:
$$\text{Effective Speech Rate} \approx N \times (1 - 0.6s)$$
On a typical podcast where $s = 0.30$ (30% silence):
- A requested speed of $N = 6.0\times$ runs the speech phonemes at only $\approx 4.9\times$!
- This gives roughly $1.1\times$ to $1.5\times$ of effective cognitive headroom.

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
