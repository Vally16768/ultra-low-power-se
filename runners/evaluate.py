def main(cfg):
    import csv, soundfile as sf, numpy as np
    sr = int(cfg.get("sample_rate", 16000))
    frame = int(sr * cfg.get("frame_ms", 20)/1000)
    hop   = int(sr * cfg.get("hop_ms", 10)/1000)
    man = cfg.get("eval", {}).get("manifest", "data/prepared/test_challenge/streaming/manifests/pairs.csv")

    # load model ...
    print(f"[streaming] frame={frame} hop={hop} manifest={man}")
    with open(man) as f:
        reader = csv.DictReader(f)
        for row in reader:
            noisy, clean = row["noisy"], row["clean"]
            x, _ = sf.read(noisy, dtype="float32"); c, _ = sf.read(clean, dtype="float32")
            # streaming windows
            out = np.zeros_like(x)
            i=0
            while i < len(x):
                seg = x[i:i+frame]
                if len(seg)<frame: seg=np.pad(seg,(0,frame-len(seg)))
                # y_seg = model(seg)  # aici vei chema inferența ta
                y_seg = seg  # stub
                out[i:i+frame] = y_seg[:min(frame, len(out)-i)]
                i += hop
            # aici poți calcula latență estimată, etc.
