import numpy as np

from speedman import io as sio


def test_wav_roundtrip(tmp_path, speechlike, sr):
    p = tmp_path / "x.wav"
    sio.save(p, speechlike, sr)
    back = sio.load(p, sr)
    assert len(back) == len(speechlike)
    assert np.corrcoef(back, speechlike)[0, 1] > 0.99


def test_eval_rate_is_16k(speechlike, sr):
    out = sio.to_eval_rate(speechlike, sr)
    assert abs(len(out) / (len(speechlike) / sr) - 16000) < 100
