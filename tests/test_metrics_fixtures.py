import numpy as np
from metrics.sisdr import sisdr
from metrics.snr_metrics import delta_snr


def test_sisdr_increases_when_noise_drops():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(16000).astype(np.float32)
    y_bad = s + 0.5 * rng.standard_normal(16000).astype(np.float32)
    y_good = s + 0.1 * rng.standard_normal(16000).astype(np.float32)
    assert sisdr(s, y_good) > sisdr(s, y_bad)


def test_delta_snr_positive_when_enhanced_better():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(16000).astype(np.float32)
    noisy = clean + rng.standard_normal(16000).astype(np.float32) * 0.5
    enhanced = clean + rng.standard_normal(16000).astype(np.float32) * 0.1
    assert delta_snr(clean, noisy, enhanced) > 0


def test_pesq_stoi_optional():
    try:
        from metrics.pesq_metric import pesq_score
        from metrics.stoi_metric import stoi_score
    except Exception:
        return  # skip dacă nu sunt instalate dependențele
    x = np.zeros(16000, dtype=np.float32)
    assert 0.0 <= stoi_score(x, x, sr=16000) <= 1.0
