"""
COPILOT DRAFT — REFERENCE ONLY. DO NOT RUN. DO NOT COPY FROM THIS FILE.

This is the "v3" script produced in the original Copilot brainstorming session,
preserved verbatim-in-substance for historical reference. It contains syntax
errors, removed librosa APIs, and concept-level DSP mistakes. Every issue is
catalogued in docs/COPILOT_CODE_REVIEW.md. The real implementation lives in
src/speedman/ and shares no code with this file.
"""

import librosa
import numpy as np
import scipy.signal as sig

# =========================
# CONFIG
# =========================

INPUT_FILE = "input.wav"
OUTPUT_FILE = "output_fast.wav"

SPEED_RATE = 10.0  # 5x, 10x, 15x
PRE_EMPHASIS_COEFF = 0.97

TRANSIENT_BOOST_DB = 3.0
TRANSIENT_WINDOW_SEC = 0.012

PAUSE_INSERT_THRESHOLD = 0.02
PAUSE_LENGTH_SEC = 0.02

ADAPTIVE_EQ_CENTER = 4500
ADAPTIVE_EQ_Q = 1.2
ADAPTIVE_EQ_MAX_GAIN_DB = 6.0

LPC_ORDER = 16

# =========================
# UTILS
# =========================

def db_to_gain(db):
    return 10 ** (db / 20.0)

def safe_normalize(y, target_peak=0.95):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    return y * (target_peak / peak)

# =========================
# STAGE 1: LOAD + PRE-EMPHASIS
# BUG (review #10): pre-emphasis with no de-emphasis -> harsh output tilt
# =========================

def load_audio(path):
    y, sr = librosa.load(path, sr=None)
    return y, sr

def pre_emphasis(y, coeff=PRE_EMPHASIS_COEFF):
    return np.append(y[0], y[1:] - coeff * y[:-1])

# =========================
# STAGE 2: ADAPTIVE EQ
# BUG (review #6): iirpeak is a band-PASS; this replaces the signal with
# its 4.5 kHz band instead of boosting it. Must be additive.
# =========================

def adaptive_clarity_eq(y, sr):
    S = np.abs(librosa.stft(y, n_fft=1024))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    high_band = (freqs > 3000) & (freqs < 8000)
    high_energy = np.mean(S[high_band])
    total_energy = np.mean(S)

    if total_energy == 0:
        gain_db = 0.0
    else:
        ratio = high_energy / total_energy
        gain_db = (0.3 - ratio) * (ADAPTIVE_EQ_MAX_GAIN_DB / 0.3)
        gain_db = np.clip(gain_db, 0.0, ADAPTIVE_EQ_MAX_GAIN_DB)

    b, a = sig.iirpeak(ADAPTIVE_EQ_CENTER / (0.5 * sr), Q=ADAPTIVE_EQ_Q)
    y_eq = sig.lfilter(b, a, y)

    return y_eq * db_to_gain(gain_db)

# =========================
# STAGE 3: LPC FORMANT PRESERVATION
# BUG (review #7): whole-file LPC is meaningless for formants (they change
# every ~20 ms); Levinson-Durbin sign convention wrong -> unstable synthesis.
# =========================

def lpc_analysis(y, order=LPC_ORDER):
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    a = np.zeros(order+1)
    e = autocorr[0]
    a[0] = 1.0

    for i in range(1, order+1):
        acc = autocorr[i]
        for j in range(1, i):
            acc -= a[j] * autocorr[i-j]
        k = acc / e
        a[i] = k
        for j in range(1, i):
            a[j] -= k * a[i-j]
        e *= (1 - k*k)

    residual = sig.lfilter(a, [1.0], y)
    return a, residual

def lpc_synthesis(residual, a):
    return sig.lfilter([1.0], a, residual)

# =========================
# STAGE 4: HYBRID WSOLA + PHASE VOCODER
# BUGS (review #4, #8): mask length mismatch crashes; blending two unaligned
# stretched signals comb-filters. Also positional librosa args (review #3).
# =========================

def transient_mask(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    mask = librosa.util.normalize(onset_env)
    mask = np.repeat(mask, 512)
    return mask[:len(y)]

def hybrid_time_compress(y, sr, rate):
    mask = transient_mask(y, sr)

    y_wsola = librosa.effects.time_stretch(y, rate)  # TypeError in librosa>=0.10

    D = librosa.stft(y)
    D_fast = librosa.phase_vocoder(D, rate)
    y_pv = librosa.istft(D_fast)

    return mask * y_wsola + (1 - mask) * y_pv  # shape mismatch

# =========================
# STAGE 5: TRANSIENT BOOSTING
# BUG (review #12, #14): runs on compressed audio; unramped gain clicks.
# =========================

def boost_transients(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    hop_length = 512
    window_samples = int(TRANSIENT_WINDOW_SEC * sr)
    gain = db_to_gain(TRANSIENT_BOOST_DB)

    y_out = y.copy()
    for o in onsets:
        start = o * hop_length
        end = min(start + window_samples, len(y_out))
        local = y_out[start:end]
        if np.max(np.abs(local)) > 0.01:
            y_out[start:end] *= gain

    return y_out

# =========================
# STAGE 6: MICRO-PAUSE INSERTION
# BUGS (review #2, #12, #13): reshape(1, -) is a syntax error; zero-splices
# click; threshold never fires on compressed audio.
# =========================

def insert_micro_pauses(y, sr):
    frame_len = int(0.03 * sr)
    pause_samples = int(PAUSE_LENGTH_SEC * sr)

    chunks = []
    for i in range(0, len(y), frame_len):
        chunk = y[i:i+frame_len]
        chunks.append(chunk)

        energy = np.mean(chunk ** 2)
        # zcr = librosa.feature.zero_crossing_rate(chunk.reshape(1, -))[0, 0]  # SyntaxError as written
        zcr = 0.0

        if energy < PAUSE_INSERT_THRESHOLD and zcr < 0.05:
            chunks.append(np.zeros(pause_samples))

    return np.concatenate(chunks)

# =========================
# STAGE 7: SPECTRAL ENVELOPE PROTECTION
# BUG (review #11): replaces magnitude with a median-blurred copy — deletes
# harmonic detail rather than protecting the envelope.
# =========================

def spectral_envelope_protect(y, sr):
    S = np.abs(librosa.stft(y))
    cep = librosa.amplitude_to_db(S)
    smooth = sig.medfilt(cep, kernel_size=(1, 7))
    S_new = librosa.db_to_amplitude(smooth)
    y_new = librosa.istft(S_new * np.exp(1j * np.angle(librosa.stft(y))))
    return y_new

# =========================
# STAGE 8: PITCH NORMALIZATION
# BUG (review #9): TSM doesn't shift pitch; there is no chipmunk effect to
# fix. This would just make normal (often female) voices unnaturally deep.
# =========================

def pitch_normalize(y, sr):
    f0, voiced = librosa.pyin(y, fmin=80, fmax=300)
    f0_mean = np.nanmean(f0)

    if f0_mean > 250:
        return librosa.effects.pitch_shift(y, sr, n_steps=-3)  # TypeError in librosa>=0.10
    return y

# =========================
# MAIN PIPELINE
# BUG (review #1): librosa.output.write_wav was removed in librosa 0.8.
# =========================

def process_file(input_path, output_path):
    y, sr = load_audio(input_path)
    y_pre = pre_emphasis(y)
    y_eq = adaptive_clarity_eq(y_pre, sr)
    a, residual = lpc_analysis(y_eq)
    residual_fast = hybrid_time_compress(residual, sr, SPEED_RATE)
    y_fast = lpc_synthesis(residual_fast, a)
    y_trans = boost_transients(y_fast, sr)
    y_paused = insert_micro_pauses(y_trans, sr)
    y_env = spectral_envelope_protect(y_paused, sr)
    y_final = pitch_normalize(y_env, sr)
    y_final = safe_normalize(y_final)
    librosa.output.write_wav(output_path, y_final, sr)  # AttributeError: removed API

if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
