# Ultra-Low-Power Speech Enhancement — Data

Acest proiect creează automat seturile de date (train/dev/test + challenge) folosind:

- **LibriSpeech (clean)** pentru voce curată
- **DEMAND/noise** + zgomote sintetice
- **VoiceBank-DEMAND** pentru test standard
- Generatorul nostru `datasets/mixgen.py` (idempotent + verify)

## 1) Cerințe sistem
  sudo apt-get update
  sudo apt-get install -y ffmpeg libsndfile1 sox parallel

Python (în venv):
pip install numpy soundfile resampy

2) Configurează căile (o singură dată)
Creează .env.local în rădăcina repo-ului:

# Locul unde ai LibriSpeech (foldere gen train-clean-100/360, dev-clean etc.)
LIBRISPEECH_ROOT=/data/LibriSpeech

# Locul unde ai DEMAND (zgomote)
DEMAND_ROOT=/data/DEMAND

# (opțional) set de RIR-uri .wav
RIRS_ROOT=/data/RIRS_NOISES
Dacă lipsesc, scripturile folosesc fallback (creează doar zgomote sintetice).

3) Un singur pas: generează totul

make datasets
La final vei avea:

data/
  clean_wav/libri/...              # Conversii mono 16k ale LibriSpeech
  noise/demand/...                 # Zgomote DEMAND (symlink sau listă)
  noise/synth/...                  # Zgomote sintetice (white/pink/brown/babble)
  lists/
    train_clean.txt
    dev_clean.txt
    noise_train.txt
    noise_dev.txt
    noise_test.txt
    noise_unseen.txt
    rir_list.txt                   # dacă există RIR-uri
    vbd_pairs_test.csv             # VoiceBank-DEMAND test pairs
  prepared/
    train_mixes/...
    dev_mixes/...
    test_challenge/...
4) Verificări & rerulare
Verificare fără regenerare:

python datasets/mixgen.py \
  --clean-list data/lists/train_clean.txt \
  --noise-list data/lists/noise_train.txt \
  --out-dir data/prepared/train_mixes \
  --snr -5 0 5 10 15 \
  --verify-only

Forțare regenerare (ignoră manifestul existent):

python datasets/mixgen.py ... --force
mixgen.py e idempotent: dacă setul există și e valid → nu mai regenerează.

5) Targets utile
make datasets – creează totul (prepare dirs, liste, conversii, mixuri)

make datasets.verify – check rapid al integrității seturilor generate

make datasets.clean – curăță rezultatele sintetice (data/prepared/*)