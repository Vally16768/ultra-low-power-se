# NISQA – MOS non-intruziv (modele pt. telephony/wb)
from nisqa.NISQA_model import NISQA
import torch
def load_nisqa(model_name: str = "NISQA_TTS_wideband") -> NISQA:
    # model zoo NISQA: NISQA_TTS_wideband, NISQA_Telecom, etc.
    m = NISQA()
    m = m.from_pretrained(model_name)
    m.eval(); return m

def nisqa_file(model: NISQA, wav_path: str) -> float:
    with torch.no_grad():
        out = model.predict_file(wav_path)  # pandas row/df
        return float(out["mos_pred"].values[0])
