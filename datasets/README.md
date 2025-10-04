Ultra-Low-Power Speech Enhancement — Data

Acest proiect creează automat seturile de date (train/dev/test + challenge) folosind:

LibriSpeech (clean) pentru voce curată

DEMAND / noise + zgomote sintetice

VoiceBank-DEMAND pentru test standard

Generatorul nostru datasets/mixgen.py (idempotent + verify)

Opțional, builder unificat care combină VoiceBank cu augmentări Libri (scripts/build_unified_dataset.py)

1) Cerințe sistem
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 sox parallel


Python (în venv):
pip install numpy soundfile resampy


Notă: pentru descărcare automată
– Kaggle CLI (VoiceBank-DEMAND): pip install kaggle + ~/.kaggle/kaggle.json
– wget pentru OpenSLR (LibriSpeech)

2) Configurează căile (o singură dată)

Creează .env.local în rădăcina repo-ului:

# Unde e LibriSpeech (foldere: train-clean-100/360, dev-clean etc.)
LIBRISPEECH_ROOT=$(PWD)/data/librispeech/LibriSpeech

# Unde e DEMAND (zgomote) — poate fi symlink
DEMAND_ROOT=$(PWD)/data/noise/demand

# (opțional) set de RIR-uri .wav
RIRS_ROOT=$(PWD)/data/rirs

# (opțional) VoiceBank-DEMAND root (detectat automat altfel)
VBD_ROOT=$(PWD)/data/voicebank-demand-16k

# (opțional) locația mixgen.py
MIXGEN=datasets/mixgen.py

3) Download rapid (VoiceBank-DEMAND + LibriSpeech)

Ai script pregătit; rulează din folderul datasets/:

# VoiceBank-DEMAND (Kaggle) + LibriSpeech (OpenSLR) + manifest VBD
./download_data.sh


Dacă vrei să verifici structura & să generezi manifestele VBD:

./verify_and_prepare.sh
# sau, prin target Makefile:
make data-verify

4) Pipeline „clasic” (mixgen) — creează train/dev/test + challenge

Un singur pas:

make datasets

Ce se produce:

data/
  clean_wav/libri/...              # Conversii mono 16k ale LibriSpeech
  noise/demand/...                 # Zgomote DEMAND (symlink sau copiate)
  noise/synth/...                  # Zgomote sintetice (white/pink/brown/babble)
  lists/
    train_clean.txt
    dev_clean.txt
    noise_train.txt
    noise_dev.txt
    noise_unseen.txt
    rir_list.txt                   # dacă există RIR-uri
    vbd_pairs_test.csv             # perechi VoiceBank test (noisy,clean,meta)
  prepared/
    train_mixes/...
    dev_mixes/...
    test_challenge/...

Verificări & rerulare

Verificare fără regenerare:

python datasets/mixgen.py \
  --clean-list data/lists/train_clean.txt \
  --noise-list data/lists/noise_train.txt \
  --out-dir data/prepared/train_mixes \
  --snr -5 0 5 10 15 \
  --verify-only


Forțare regenerare (ignoră manifestul existent):

python datasets/mixgen.py ... --force

mixgen.py este idempotent: dacă setul există și e valid → nu mai regenerează.

5) Builder „unificat” (VBD ∪ augment Libri)

Alternativ la pipeline-ul clasic, poți construi un set final pentru antrenare/validare într-un singur loc:

# doar VoiceBank (fără augment Libri)
python scripts/build_unified_dataset.py --mix_per_clean 0

# cu augmentare (2 mixuri/clean, SNR {0,5,10,15})
python scripts/build_unified_dataset.py --mix_per_clean 2 --snr 0,5,10,15

# dacă nu ai încă zgomote, permite generare automată de zgomote sintetice:
python scripts/build_unified_dataset.py --mix_per_clean 2 --snr 0,5,10,15 --auto_synth 1


Layout rezultat:

data/datasets/final/
  vbd_train.csv            # perechi VoiceBank (train)
  vbd_test.csv             # perechi VoiceBank (test)
  aug.csv                  # perechi augment Libri+noise
  train.csv                # vbd_train ∪ aug
  val.csv                  # vbd_test
  aug/noisy/*.wav          # mixturi generate
  meta.json                # configurare & număr perechi


Wrapper cu debug (opțional):

datasets/build_unified_dataset.sh

6) Antrenare rapidă

Pe VoiceBank standard:

python runners/train.py \
  --train_csv data/voicebank-demand-16k/train.csv \
  --val_csv   data/voicebank-demand-16k/test.csv \
  --sr 16000 --epochs 50 --batch_size 8 \
  --out artifacts/exp/vbd_16k_mamba_unet


Pe setul unificat:

python runners/train.py \
  --train_csv data/datasets/final/train.csv \
  --val_csv   data/datasets/final/val.csv \
  --sr 16000 --epochs 50 --batch_size 8 \
  --out artifacts/exp/unified_vbd_libri_aug


Sau cu Makefile (config din YAML controlează tot):

make train

7) Target-uri utile (Makefile)
make datasets          # pipeline complet (prepare dirs/lists & generate mixes)
make datasets.verify   # check rapid al integrității seturilor generate
make datasets.clean    # curăță doar seturile sintetice (data/prepared/*)
make data-train        # reconstruiește manifestele VoiceBank (train/test)
make data-verify       # verifică VoiceBank + Libri (manifest & counts)
make dataset-unified   # construiește data/datasets/final/{train,val}.csv

8) Troubleshooting

aug: 0 în builderul unificat: nu s-au găsit zgomote în data/noise/{demand,custom,dev,unseen,synth}.
Soluții:

python make_synth_noises.py (generează data/noise/synth/), sau

ln -s /path/DEMAND data/noise/demand.

Kaggle CLI afișează License(s): unknown: nu e eroare; autorul datasetului nu a setat licența în pagina Kaggle. Descărcarea e OK.

Rate/Canale: totul trebuie să fie mono 16k. Conversia Libri se face cu scripts/prepare_librispeech.sh.

9) Structură recomandată finală
data/
  librispeech/LibriSpeech/...
  clean_wav/libri/...              # WAV 16k
  voicebank-demand-16k/
    clean_trainset_28spk_wav/
    noisy_trainset_28spk_wav/
    clean_testset_wav/
    noisy_testset_wav/
    train.csv
    test.csv
  noise/
    demand/...                     # opțional (symlink)
    synth/...
    custom/...                     # opțional
    dev/, unseen/                  # opțional
  lists/...
  prepared/...
  datasets/final/...
