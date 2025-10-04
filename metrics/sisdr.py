import numpy as np


def sisdr(s, s_hat, eps=1e-8):
    s = s.astype(np.float32).reshape(-1)
    s_hat = s_hat.astype(np.float32).reshape(-1)
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)
    alpha = (np.dot(s_hat, s) + eps) / (np.dot(s, s) + eps)
    s_target = alpha * s
    e_noise = s_hat - s_target
    num = np.sum(s_target**2) + eps
    den = np.sum(e_noise**2) + eps
    return 10.0 * np.log10(num / den)
