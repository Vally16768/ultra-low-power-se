from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow import keras

def plot_history(history_obj: keras.callbacks.History, out_dir: Path):
    hist = history_obj.history
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_pair(y1, y2=None, title=None, fname=None, ylabel=None):
        if y1 not in hist and (not y2 or y2 not in hist):
            return
        plt.figure(figsize=(7, 4))
        if y1 in hist: plt.plot(hist[y1], label=y1)
        if y2 and y2 in hist: plt.plot(hist[y2], label=y2)
        plt.xlabel("epoch")
        plt.ylabel(ylabel or y1)
        plt.title(title or (ylabel or y1))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120)
        plt.close()

    plot_pair("loss", "val_loss", "Loss vs Val Loss", "loss_val_loss.png", "Loss")
    plot_pair("mae", "val_mae", "MAE vs Val MAE", "mae_val_mae.png", "MAE")

    if "si_snr_tf" in hist or "val_si_snr_tf" in hist:
        plot_pair("si_snr_tf", "val_si_snr_tf", "SI-SNR vs Val SI-SNR", "sisnr_val_sisnr.png", "SI-SNR (dB)")

    key = "learning_rate" if "learning_rate" in hist else ("lr" if "lr" in hist else None)
    if key:
        plt.figure(figsize=(7, 4))
        plt.plot(hist[key])
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "learning_rate.png", dpi=120)
        plt.close()
