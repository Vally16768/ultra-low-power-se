# o singură dată / când adaugi date noi
./scripts/prepare_dirs.sh
./scripts/prepare_librispeech.sh
./scripts/prepare_noises.sh
./scripts/build_lists.sh

# generează train/dev + challenge (după ce ai zgomote)
./scripts/gen_mixes.sh

# test standard VoiceBank-DEMAND
./scripts/build_vbd_pairs.sh
