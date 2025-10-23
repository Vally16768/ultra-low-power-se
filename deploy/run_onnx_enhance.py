# run_onnx_enhance.py

#!/usr/bin/env python3
import argparse, numpy as np, onnxruntime as ort, soundfile as sf, librosa
from pathlib import Path

def frame_audio(x, frame, hop):
    if len(x) <= frame:  # pad once
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
    # compensate overlap
    wsum = np.zeros(total, dtype=np.float32)
    for i in range(len(frames)):
        s = i*hop
        wsum[s:s+frame] += win
    wsum = np.maximum(wsum, 1e-6)
    return y/wsum

def enhance_file(sess, wav_in, wav_out, sr=16000, frame_s=2.0, hop_s=1.0):
    x, inp_sr = sf.read(wav_in)
    if x.ndim == 2: x = x.mean(axis=1)
    if inp_sr != sr:
        x = librosa.resample(x.astype(np.float32), orig_sr=inp_sr, target_sr=sr)
    x = x.astype(np.float32)

    frame = int(frame_s*sr)
    hop   = int(hop_s*sr)
    X = frame_audio(x, frame, hop)  # [N, T]
    Xb = X[:, :, None]              # [N, T, 1]

    outs = []
    for i in range(len(Xb)):
        y = sess.run([sess.get_outputs()[0].name], {"input:0": Xb[i:i+1]})[0]
        outs.append(y[0, :, 0])
    Y = np.stack(outs, 0)
    y = ola(Y, hop)

    Path(wav_out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_out, y, sr)
    return len(x)/sr, len(y)/sr

def main():
    ap = argparse.ArgumentParser("Run ONNX enhancer on WAV(s)")
    ap.add_argument("--onnx", default="artifacts/tf_manifest_only/model.onnx")
    ap.add_argument("--in_path", required=True, help="WAV file or directory")
    ap.add_argument("--out_dir", default="artifacts/onnx_out")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_s", type=float, default=2.0)
    ap.add_argument("--hop_s", type=float, default=1.0)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_path = Path(args.in_path)
    wavs = [in_path] if in_path.is_file() else list(in_path.rglob("*.wav"))
    assert wavs, f"No WAVs under {in_path}"

    for w in wavs:
        rel = w.name if in_path.is_file() else w.relative_to(in_path)
        out = Path(args.out_dir)/rel
        out.parent.mkdir(parents=True, exist_ok=True)
        t_in, t_out = enhance_file(sess, str(w), str(out), args.sr, args.frame_s, args.hop_s)
        print(f"[OK] {w} -> {out}  (sec in/out: {t_in:.2f}/{t_out:.2f})")

if __name__ == "__main__":
    main()

