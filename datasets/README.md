# datasets – Mixers & Loaders

- `mixgen.py`: deterministic mix generation for challenge sets.
- `librispeech_mix.py`, `voicebank_demand.py`, `voxceleb_eval.py`: dataset loaders.
- `transforms.py`: audio transforms.

### Data prep (example)
```bash
bash scripts/build_vbd_pairs.sh   # builds VoiceBank-DEMAND pairs lists
python datasets/mixgen.py --help # see parameters
```
Expected: lists & synthetic JSON metadata in `data/lists` and `data/prepared/...`.
