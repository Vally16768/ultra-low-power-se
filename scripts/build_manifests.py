# scripts/build_manifests.py
import csv, pathlib, random

LS_ROOT = pathlib.Path("data/librispeech/LibriSpeech")
VB_ROOT = pathlib.Path("data/voicebank-demand-16k")

def list_flacs_or_wavs(root):
    return sorted([*root.rglob("*.flac"), *root.rglob("*.wav")])

def main():
    random.seed(0)

    # --- LibriSpeech: doar CLEAN ---
    train_clean = list_flacs_or_wavs(LS_ROOT / "train-clean-100") + \
                  list_flacs_or_wavs(LS_ROOT / "train-clean-360")
    dev_clean   = list_flacs_or_wavs(LS_ROOT / "dev-clean")
    test_clean  = list_flacs_or_wavs(LS_ROOT / "test-clean")

    (pathlib.Path("manifests")).mkdir(exist_ok=True)

    with open("manifests/train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","path_clean"])
        for p in train_clean:
            w.writerow(["train", p.as_posix()])

    with open("manifests/test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","path_clean"])
        for p in test_clean:
            w.writerow(["test", p.as_posix()])

    # --- VoiceBank-DEMAND: validation/test pairs ---
    vb_clean_test  = sorted((VB_ROOT / "clean_testset_wav").rglob("*.wav"))
    vb_noisy_test  = sorted((VB_ROOT / "noisy_testset_wav").rglob("*.wav"))
    assert len(vb_clean_test) == len(vb_noisy_test)

    with open("manifests/val_voicebank.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_noisy","path_clean"])
        for p_noisy, p_clean in zip(vb_noisy_test, vb_clean_test):
            # potrivire după nume de fișier — în distribuțiile standard se aliniază 1-la-1
            w.writerow([p_noisy.as_posix(), p_clean.as_posix()])

    print("Done: manifests/train.csv, manifests/test.csv, manifests/val_voicebank.csv")

if __name__ == "__main__":
    main()
