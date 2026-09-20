"""Modulation-spectrum analysis: an intelligibility correlate that survives past 3.5x.

WHY THIS AND NOT STOI. STOI correlates the temporal envelopes of a reference and a
degraded signal, which requires them to be the same length and time-aligned. Speedman's
output is 1/N the length and non-uniformly warped, so there is no aligned comparison to
make. More fundamentally, STOI models intelligibility lost to noise and reverberation,
not to rate: a clean 6x stretch would score well on it while being unintelligible, because
the limit at 6x is auditory temporal integration, not spectral distortion. STOI would
mostly be measuring how good rubberband is.

WHY MODULATION FREQUENCY IS THE RIGHT AXIS. Speech intelligibility depends on the temporal
envelope modulations in roughly 2-16 Hz, peaking near the 4-5 Hz syllabic rate; this is the
basis of the Speech Transmission Index (IEC 60268-16). Compressing time by N shifts every
modulation frequency up by N. At 5x the 4 Hz syllabic peak lands near 20 Hz, outside the
band the auditory system integrates for speech. That shift IS the mechanism by which fast
speech stops being intelligible, so measuring it measures the thing that matters -- and it
needs no reference alignment, so it works at any speed.

WHAT SPEEDMAN'S CLAIM PREDICTS. Because pauses absorb more than their share of the
compression, speech itself is compressed by less than N. So at the same nominal speed,
Speedman's output should retain more modulation energy in the syllabic band than a uniform
stretch does, and its modulation peak should sit at a lower frequency. That is a concrete,
falsifiable prediction, and it is `effective_speech_rate` measured on the actual output
signal rather than assumed from a formula.

WHAT THIS IS NOT. A correlate, not a verdict. It cannot say "6x is intelligible". It can
say "Speedman preserves more syllabic-rate modulation at 6x than uniform does". Trust it
only to the extent it agrees with WER where both work -- see `validate()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

# Octave bands from the STI standard that fit below Nyquist at 24 kHz.
OCTAVE_CENTRES_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

ENVELOPE_SR = 200.0  # plenty for modulation content below 100 Hz

SYLLABIC_LO_HZ = 2.0
SYLLABIC_HI_HZ = 16.0
"""The band that carries speech intelligibility. 4-5 Hz is the syllabic rate; below 2 Hz
is phrase-level prosody and above 16 Hz the auditory system no longer integrates it as
speech rhythm."""

ANALYSIS_LO_HZ = 0.5
ANALYSIS_HI_HZ = 80.0


@dataclass(frozen=True)
class ModulationStats:
    syllabic_fraction: float
    """Share of modulation energy inside 2-16 Hz. Higher is better."""
    peak_hz: float
    """Modulation frequency carrying the most energy. ~4-5 Hz for unprocessed speech."""
    centroid_hz: float
    """Energy-weighted mean modulation frequency. Moves up as audio is compressed."""

    def as_dict(self) -> dict:
        return {
            "syllabic_fraction": self.syllabic_fraction,
            "peak_hz": self.peak_hz,
            "centroid_hz": self.centroid_hz,
        }


def _band_envelope(y: np.ndarray, sr: int, centre: float) -> np.ndarray | None:
    """Envelope of one octave band, decimated to ENVELOPE_SR.

    Returns None when the band does not fit below Nyquist.
    """
    lo, hi = centre / np.sqrt(2.0), centre * np.sqrt(2.0)
    nyq = sr / 2.0
    if hi >= nyq * 0.99:
        return None

    sos = signal.butter(4, [lo / nyq, min(hi, nyq * 0.98) / nyq], btype="band", output="sos")
    band = signal.sosfiltfilt(sos, y)

    # Hilbert magnitude is the standard envelope here; rectify-and-smooth would add a
    # lowpass whose cutoff then has to be justified against the modulation band we measure.
    env = np.abs(signal.hilbert(band))

    step = max(1, int(round(sr / ENVELOPE_SR)))
    # Anti-alias before decimating: envelope content above ENVELOPE_SR/2 would fold back
    # into the syllabic band and inflate exactly the number we care about.
    sos_lp = signal.butter(4, (ENVELOPE_SR / 2.0) / (sr / 2.0), btype="low", output="sos")
    env = signal.sosfiltfilt(sos_lp, env)
    return env[::step]


def analyse(y: np.ndarray, sr: int) -> ModulationStats:
    """Modulation statistics of a signal, averaged over octave bands."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < sr:  # under a second: nothing meaningful at 2 Hz resolution
        return ModulationStats(float("nan"), float("nan"), float("nan"))

    spectra = []
    freqs = None
    for centre in OCTAVE_CENTRES_HZ:
        env = _band_envelope(y, sr, centre)
        if env is None or env.size < 64:
            continue
        env = env - env.mean()  # DC would dominate and is not modulation
        if not np.any(env):
            continue

        nperseg = min(len(env), int(ENVELOPE_SR * 8))  # 8s window -> 0.125 Hz resolution
        f, pxx = signal.welch(env, fs=ENVELOPE_SR, nperseg=nperseg)
        # Normalise per band so loud bands do not dominate the average.
        total = pxx.sum()
        if total > 0:
            spectra.append(pxx / total)
            freqs = f

    if not spectra or freqs is None:
        return ModulationStats(float("nan"), float("nan"), float("nan"))

    mean_spec = np.mean(spectra, axis=0)

    window = (freqs >= ANALYSIS_LO_HZ) & (freqs <= ANALYSIS_HI_HZ)
    syllabic = (freqs >= SYLLABIC_LO_HZ) & (freqs <= SYLLABIC_HI_HZ)

    denom = mean_spec[window].sum()
    frac = float(mean_spec[syllabic].sum() / denom) if denom > 0 else float("nan")

    wf, ws = freqs[window], mean_spec[window]
    peak = float(wf[int(np.argmax(ws))]) if ws.size else float("nan")
    centroid = float((wf * ws).sum() / ws.sum()) if ws.sum() > 0 else float("nan")

    return ModulationStats(syllabic_fraction=frac, peak_hz=peak, centroid_hz=centroid)
