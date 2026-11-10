# data/

Datasets live here and are not committed (see `.gitignore`).

Planned layout, one folder per word with 1-second 16 kHz mono WAV clips, the
same layout Speech Commands uses:

```
data/
├── speech_commands_v0.02/   # Google Speech Commands v2 (on, off, unknown words, background noise)
├── mswc_en/                 # clips picked from the Multilingual Spoken Words Corpus
├── recorded/                # our own recordings: lamp/, bright/, dim/, on/, off/
└── synthetic/               # text-to-speech clips, if we use them
```
