# data/

Datasets live here and are not committed (see `.gitignore`). `notebooks/01_prepare_data.ipynb`
fills this folder.

```
data/
├── downloads/               # the Speech Commands archive as downloaded
├── speech_commands_v0.02/   # Google Speech Commands v2: one folder per word, plus _background_noise_
└── mswc/
    ├── metadata.json.gz     # MSWC word list with clip names
    ├── en_splits_subset.csv # MSWC train/dev/test split for the words we use
    └── clips/<word>/*.wav   # MSWC clips converted to 16 kHz mono WAV, one second each
```

All clips are one-second 16 kHz mono WAV files in one folder per word.
