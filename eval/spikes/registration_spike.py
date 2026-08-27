"""Time-map registration spike.

Answers: can the stretching backend place a non-uniform time map precisely
enough for transient protection to be mechanically possible?

Results and interpretation: docs/SPIKE_RESULTS.md
This is a permanent fixture -- re-run on any change of engine (R2/R3), sample
rate, anchor strategy or backend. Registration error is a property of the
stretching engine and every lever-2 parameter is downstream of it.

Deps: numpy scipy soundfile pyrubberband + system rubberband-cli
"""
import numpy as np
import pyrubberband as pyrb
from scipy.signal import butter, sosfiltfilt, find_peaks

SR = 24000
rng = np.random.default_rng(0)


def build():
    """Alternating 0.8 s speech / 0.2 s pause, with a 4 ms 6 kHz marker burst
    at a known offset inside each speech block."""
    blocks, marks, t = [], [], 0.0
    sos = butter(4, 3000, "lp", fs=SR, output="sos")
    for _ in range(10):
        n = int(0.8 * SR)
        sp = sosfiltfilt(sos, rng.normal(0, 1, n)) * 0.20
        sp += 0.15 * np.sin(2 * np.pi * 180 * np.arange(n) / SR)
        off, ln = int(0.20 * SR), int(0.004 * SR)
        sp[off:off + ln] += np.hanning(ln) * np.sin(2 * np.pi * 6000 * np.arange(ln) / SR) * 1.2
        marks.append(t + 0.202)
        blocks.append(sp)
        t += 0.8
        blocks.append(np.zeros(int(0.2 * SR)))
        t += 0.2
    return np.concatenate(blocks).astype(np.float32), np.array(marks)


def profile(n, N):
    """Per-sample instantaneous rate implementing both levers, normalised so the
    whole file averages to exactly N."""
    r = np.ones(n)
    blk = int(SR)
    for i in range(10):
        s = i * blk
        e = s + int(0.8 * SR)
        r[s:s + int(0.16 * SR)] = 0.7                    # transient (protected)
        r[s + int(0.16 * SR):s + int(0.48 * SR)] = 2.0   # vowel centre (crushed)
        r[s + int(0.48 * SR):e] = 1.0                    # remainder
        r[e:s + blk] = 2.5                               # pause
    return r * (np.sum(1.0 / r) / (n / N))


def tmap(rate, n, anchor_ms):
    """Integrate 1/rate to output position; sample anchors; pin the final anchor
    and force strict monotonicity. This is what gives exact duration."""
    op = np.concatenate([[0.0], np.cumsum(1.0 / rate)])[:n + 1]
    step = max(1, int(anchor_ms * SR / 1000))
    tm = []
    for i in list(range(0, n, step)) + [n]:
        b = int(round(op[i]))
        if tm and b <= tm[-1][1]:
            b = tm[-1][1] + 1
        tm.append((int(i), b))
    return tm


def detect(y, k=0.45):
    env = np.abs(sosfiltfilt(butter(6, [5400, 6600], "bp", fs=SR, output="sos"), y))
    env = sosfiltfilt(butter(2, 150, "lp", fs=SR, output="sos"), env)
    pk, _ = find_peaks(env, height=np.max(env) * k, distance=int(0.02 * SR))
    return pk / SR


def main(rbargs=None):
    sig, mk = build()
    n = len(sig)
    print(f"input {n/SR:.1f}s, {len(mk)} markers, rbargs={rbargs}\n")
    print(f"{'N':>3} {'anchor':>7} {'dur err':>9} {'mean':>8} {'max':>8} {'drift':>10} {'jitter':>8}")
    print("-" * 60)
    for N in (5, 8, 10):
        rate = profile(n, N)
        for ms in (20, 10, 5, 2):
            tm = tmap(rate, n, ms)
            y = pyrb.timemap_stretch(sig, SR, tm, rbargs=rbargs)
            ax = np.array([a for a, _ in tm])
            ay = np.array([b for _, b in tm])
            want = np.interp(mk * SR, ax, ay) / SR
            got = detect(y)
            if len(got) != len(mk):
                got = np.array([got[np.argmin(np.abs(got - w))] for w in want])
            err = (got - want) * 1000
            fit = np.polyfit(want, err, 1)
            dur_err = (len(y) - n / N) / (n / N) * 100
            print(f"{N:>3} {ms:>5}ms {dur_err:>8.2f}% {np.mean(np.abs(err)):>7.2f}ms "
                  f"{np.max(np.abs(err)):>7.2f}ms {fit[0]:>7.2f}ms/s "
                  f"{np.std(err - np.polyval(fit, want)):>7.2f}ms")
        print()


if __name__ == "__main__":
    main()
    # engine comparison: main(rbargs={'-3': ''})
