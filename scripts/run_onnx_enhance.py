#!/usr/bin/env python3
import argparse, numpy as np, onnxruntime as ort, soundfile as sf, librosa, os
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

def pick_io(sess):
    in0 = sess.get_inputs()[0].name
    out0 = sess.get_outputs()[0].name
    return in0, out0

def load_audio_any(path, target_sr):
    """Read with soundfile if possible, else fallback to librosa.load."""
    ext = Path(path).suffix.lower()
    try:
        x, sr = sf.read(path, always_2d=False)
        if x.ndim == 2: x = x.mean(axis=1)
        x = x.astype(np.float32)
        if sr != target_sr:
            x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return x, sr
    except Exception:
        # soundfile cannot handle some compressed formats (e.g. mp3) in some setups
        x, sr = librosa.load(path, sr=target_sr, mono=True)
        x = x.astype(np.float32)
        return x, sr

def frame_audio(x, frame, hop):
    if len(x) <= frame:
        pad = frame - len(x)
        x = np.pad(x, (0, pad), mode="reflect")
    frames = []
    for i in range(0, len(x) - frame + 1, hop):
        frames.append(x[i:i+frame])
    tail = (len(x) - frame) % hop
    if tail != 0:
        start = len(x) - frame
        frames.append(x[start:start+frame])
    return np.stack(frames, 0)

def ola(frames, hop):
    frame = frames.shape[1]
    total = hop*(len(frames)-1) + frame
    y = np.zeros(total, dtype=np.float32)
    win = np.hanning(frame).astype(np.float32)
    for i, fr in enumerate(frames):
        s = i*hop
        y[s:s+frame] += fr * win
    wsum = np.zeros(total, dtype=np.float32)
    for i in range(len(frames)):
        s = i*hop
        wsum[s:s+frame] += win
    wsum = np.maximum(wsum, 1e-6)
    return y/wsum

def enhance_file(sess, in_name, out_name, wav_in, wav_out, sr=16000, frame_s=2.0, hop_s=1.0, batch=1):
    x, _ = load_audio_any(wav_in, sr)
    frame = int(frame_s*sr)
    hop   = int(hop_s*sr)
    X = frame_audio(x, frame, hop)        # [N, T]
    Xb = X[:, :, None]                    # [N, T, 1]

    outs = []
    for i in range(0, len(Xb), batch):
        chunk = Xb[i:i+batch]             # [B, T, 1]
        y = sess.run([out_name], {in_name: chunk})[0]  # [B, T, 1]
        outs.append(y[..., 0])
    Y = np.concatenate(outs, axis=0)      # [N, T]
    y = ola(Y, hop)

    Path(wav_out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_out, y, sr)
    return len(x)/sr, len(y)/sr

def gather_audio(root: Path):
    if root.is_file() and root.suffix.lower() in AUDIO_EXTS:
        return [root]
    if root.is_dir():
        return [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return []

def main():
    ap = argparse.ArgumentParser("Run ONNX enhancer on audio file(s)")
    ap.add_argument("--onnx", default="artifacts/tf_manifest_only/onnx/unet1d_fp32.onnx")
    ap.add_argument("--in_path", required=True, help="Audio file or directory (wav/flac/mp3/ogg/...)")
    ap.add_argument("--out_dir", default="artifacts/onnx_out")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_s", type=float, default=2.0)
    ap.add_argument("--hop_s", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1, help="Batch frames for speed")
    ap.add_argument("--provider", default="CPUExecutionProvider",
                    help="ONNX Runtime EP (e.g., CPUExecutionProvider, CUDAExecutionProvider)")
    args = ap.parse_args()

    root = Path(args.in_path)
    wavs = gather_audio(root)
    if not wavs:
        print(f"[SKIP] No audio found under '{root}'. Supported: {sorted(AUDIO_EXTS)}")
        return

    sess = ort.InferenceSession(args.onnx, providers=[args.provider])
    in_name, out_name = pick_io(sess)

    for src in wavs:
        # Mirror relative path, force .wav extension
        rel = src.name if root.is_file() else src.relative_to(root)
        out = Path(args.out_dir)/rel
        out = out.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)

        t_in, t_out = enhance_file(sess, in_name, out_name, str(src), str(out),
                                   args.sr, args.frame_s, args.hop_s, args.batch)
        print(f"[OK] {src} -> {out}  (sec in/out: {t_in:.2f}/{t_out:.2f})")

if __name__ == "__main__":
    main()
