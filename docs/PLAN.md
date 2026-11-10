# Project plan: voice-controlled fairy light lamp

Goal: a fairy light lamp that responds to spoken commands ("crystal on", "crystal off",
"crystal bright", …) recognised on the microcontroller itself, with no phone or cloud.
The model is trained in Jupyter with TensorFlow and runs on the board with
TensorFlow Lite for Microcontrollers (TFLM).

## 1. Decisions to make first

### 1.1 Board: it must be the Nano 33 BLE *Sense*

The plain **Arduino Nano 33 BLE has no microphone**. The **Nano 33 BLE Sense**
(Rev1 or Rev2) has the on-board PDM microphone that the tinyMLx `micro_speech`
example reads. Same processor on both (nRF52840, Cortex-M4F, 1 MB flash, 256 kB RAM).
If the board on hand is the plain BLE, it needs an external PDM/I2S microphone
breakout and a different audio provider.

### 1.2 How to recognise two-word commands

The reference colabs classify single one-second words. "Crystal on" is two words, and
there are two ways to handle that:

| | A. Single words + state machine (recommended) | B. Whole phrase as one class |
|---|---|---|
| Model classes | silence, unknown, `crystal`, `on`, `off`, `bright` | silence, unknown, `crystal on`, `crystal off`, `crystal bright`, … |
| Audio window | 1 s, 49×40 features (unchanged from `micro_speech`) | 1.5–2 s, about twice the input and compute |
| Public data | `on`/`off` in Speech Commands; `crystal`/`bright` in MSWC | none; every clip must be recorded or synthesised |
| Firmware | hear `crystal`, then accept `on`/`off`/`bright` for about 2 s | direct mapping, no state machine |
| False triggers | low: two detections in a row are needed | depends entirely on the custom data |
| Adding a command | one new word | one new phrase, new recordings |

Decision: **A**. It reuses the proven feature pipeline and public
data, and `crystal` acts as a wake word, which matters because short words like "on"
are easily confused with "one" or "off" in normal conversation. B stays open as a
later experiment once there is a recording workflow.

### 1.3 Wake word: `crystal`

The first models used `lamp` as the wake word. MSWC has only 292 clips of it (30
in the test split), and the models confused it with *land*, *camp* and *plan*.
`crystal` replaces it: 2,003 clips in MSWC, two syllables with a distinctive
"kr-st" cluster, rare in everyday speech, unlike any of the commands, and a fit
for a glass globe lamp. The `lamp` models and reports are kept in `models/lamp/`
and `reports/lamp/` for comparison. Other candidates checked: `star` (2,127
clips, but one syllable and close to *start*), `magic` (462), `lantern` (345),
`fairy` (137).

### 1.4 Train in TensorFlow 2, not with the TF 1.15 scripts

The three tinyMLx colabs call TensorFlow's `speech_commands/train.py` under
TensorFlow 1.15, which no longer installs on current Python or Colab. This project
re-implements the same pipeline in TensorFlow 2 / Keras, keeping the two things
that must match the firmware:

- the **micro front end** features (16 kHz, 30 ms window, 20 ms stride, 40 bins,
  49 slices), available as a Python op in the TensorFlow pip package;
- **full int8 quantisation** of the model.

Both are checked on this machine by `notebooks/00_environment_check.ipynb`
(TensorFlow 2.21, Apple Silicon, CPU).

## 2. Datasets

| Source | What it gives us | Licence | Notes |
|---|---|---|---|
| [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands) | `on`, `off`; 33 other words for the "unknown" class; background noise clips | CC BY 4.0 | About 2.4 GB, roughly 105,000 one-second clips, 35 words. Does **not** contain `crystal`, `lamp`, `bright`, `dim` or `light`. |
| [Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/datasets/multilingual-spoken-words/) | `crystal`, `bright`, more `off`, and similar-sounding words for the unknown class | CC BY 4.0 | One-second clips cut from Common Voice sentences. English audio is a single 35 GB archive, which `fairy_kws.mswc` streams once, keeping only the words we ask for. No words shorter than three letters, so no `on`. |
| Own recordings | All keywords, in the voices and rooms that matter | ours | The tinyMLx custom-dataset colab uses the Open Speech Recording web tool. Recording through the board's own microphone gives the best match to deployment. Target: 50–100 clips per word per speaker, several speakers. |
| Synthetic speech (TTS) | Hundreds of voices for any word, cheaply | depends on the voice model | The approach used by the [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) project (Piper TTS plus noise and room augmentation). Good for volume; real recordings are still needed for validation. |

Clips available per word (checked against the MSWC metadata):

| Word | Speech Commands | MSWC | Verdict |
|---|---|---|---|
| `on` | 3,845 | not included | fine |
| `off` | 3,745 | 6,511 | fine |
| `bright` | none | 1,733 | fine |
| `crystal` | none | 2,003 | fine |
| `lamp` | none | 292 | thin; the first wake word, replaced by `crystal` |
| `dim` | none | 74 | too few to train on |
| `low` / `dark` / `light` | none | 1,740 / 1,557 / 2,998 | available if a fourth command is wanted |

Vocabulary: `crystal` (wake word), `on`, `off`, `bright`. A
counterpart to `bright` is left open: `dim` would need our own recordings or
synthetic speech, while `low` or `dark` could be added from MSWC today.

Own recordings remain the way to build a test set that reflects real use and to
close the gap between dataset audio and the board's microphone; TTS only if a word is still short of data.

## 3. Model

| Model | Size (int8) | Role |
|---|---|---|
| `tiny_conv` (one conv layer + dense, from `micro_speech`) | 27 kB, about 0.34 million multiply-adds | Baseline. Runs unchanged in the reference firmware. |
| `ds_cnn` (depthwise-separable CNN, after the MLPerf Tiny keyword spotting reference) | 37 kB, about 3.7 million multiply-adds | Step up in accuracy at roughly ten times the computation; its speed on the board is not yet measured. |

Board budget: 1 MB flash and 256 kB RAM leave room for either. Targets to confirm
on hardware: model under 100 kB, and inference fast enough that the recogniser
gets several classifications per second to average over.

Metrics, in order of importance for a lamp that sits in a living room:

1. **False activations per hour** on continuous non-command audio (speech, TV, music).
2. Per-word recall on the held-out recorded test set.
3. Confusion between `on` and `off`.
4. Size and latency on the board.

## 4. Phases

### Phase 0: Tooling (done in this setup)
- Git repository layout, uv environment, JupyterLab, lint, tests, CI workflow.
- Environment check notebook: features, int8 conversion, C array export.

### Phase 1: Data (done for public data)
- `01_prepare_data.ipynb`: download Speech Commands, check MSWC word counts, fetch
  the MSWC clips, build the clip table with speaker-separated splits, listen, plot
  features.
- Still to do: record our own clips of all four words.

### Phase 2: Train and convert (first version done)
- `02_train_keywords.ipynb`: train `tiny_conv` and `ds_cnn` on the full vocabulary
  with balanced classes, time shift and background noise.
- Convert to int8, compare float and int8 accuracy, per-word results, detection
  threshold table, export `.tflite`, C array and label list.
- First results with the wake word `lamp` (int8 models, test set with background
  noise, speakers not seen in training): `ds_cnn` 88% accuracy, `tiny_conv` 78%. Main weaknesses: MSWC
  words that sound like keywords are rejected only 74% of the time by `ds_cnn`
  (*light* is taken for `bright`, *land* for `lamp`), and `lamp` has just 30 test
  clips and 70% recall. Not yet tested on the board or on our own voices.
- Results with the wake word `crystal` (same setup): `ds_cnn` 90% accuracy,
  `tiny_conv` 80%. `ds_cnn` recall: `crystal` 95% (201 test clips), `on` 92%,
  `off` 91%, `bright` 94%.
- Wake word comparison, `ds_cnn`, both models scored on the same 15,894 test
  clips of other words:

  | | `lamp` | `crystal` |
  |---|---|---|
  | Wake word recall at a score threshold of 0.8 | 53% | 92% |
  | False wakes at 0.8, Speech Commands words | 0.28% | 0.03% |
  | False wakes at 0.8, MSWC words (mostly chosen sound-alikes) | 1.8% | 3.2% |
  | Wake word recall when the threshold allows 0.5% false wakes | 43% | 83% |

  `crystal` is heard far more reliably and almost never fires on ordinary
  words. What still triggers it are words with the same "s-t" pattern:
  *Christian*, *system*, *sister*, *Bristol*, *coastal*. The pool holds 36
  sound-alikes of `crystal` and 10 of `lamp`, so the MSWC row is harder on
  `crystal`.
- Weakest spot now: `bright` precision (61% for `ds_cnn`), mostly *light* and
  *right* taken for `bright`. Still not tested on the board or on our own voices.

### Phase 2b: Live test on the computer (script done, real voices to do)
- `uv run python -m fairy_kws.listen` runs the int8 model on the computer's
  microphone with the recogniser the firmware will need: scores averaged over
  0.3 s, threshold 0.7, wake word then one command within 2 s.
- On simulated streams (a test clip of `crystal`, a pause, a test clip of a
  command, with background noise) `ds_cnn` completes about 8 of 10 commands; a
  1.5 s window proved too short once the pause reached 0.6 s. Of single other
  words, about 3% trigger the wake word, and 1 of 400 pairs of other words
  changed the lamp.
- To do: try it with our own voices and note which words fail.

### Phase 3: Baseline on the board
- Install Arduino IDE or `arduino-cli`, the Nano 33 BLE core and the
  `Harvard_TinyMLx` library.
- Flash stock `micro_speech`, then swap in our model, labels and category count.
- **Feature parity test**: print the features the board computes and compare with
  the notebook's features for similar audio. This is the most common place for a
  model to work in the notebook and fail on the device.
- `on`/`off` drive the built-in LED.

### Phase 4: Make it reliable
- Add our own recordings and retrain; test on recordings made with the board.
- Streaming test on long negative audio (speech, TV, music): false activations
  per hour; tune the detection threshold and averaging window.
- Decide between `tiny_conv` and `ds_cnn` from accuracy and measured on-board speed.

### Phase 5: Lamp behaviour and hardware
- Firmware state machine: `crystal` opens a ~2 s window; `on`/`off`/`bright`
  act only inside it.
- Drive the LED string through a logic-level MOSFET with PWM (the board's pins are
  3.3 V and cannot power the string directly); brightness levels for `bright`.
- Visual feedback when the wake word is heard (short pulse of the lights).

### Phase 6: Towards a product
- Microphone placement in the wooden base, acoustic opening, noise from the LED
  driver.
- Power: USB vs battery, idle current while always listening.
- Robustness testing with more speakers, accents, distances, background noise.
- Cost-down options once the model is fixed (bare nRF52840 module or another
  Cortex-M4 with a PDM microphone, custom PCB).

## 5. Risks

| Risk | Mitigation |
|---|---|
| `crystal` and `bright` come only from MSWC, `on` only from Speech Commands, so the model may learn recording style along with the word | Background noise mixed into every clip; own recordings of all words next |
| Training and device features differ | Same micro front end on both sides; parity test in Phase 3 |
| False activations from ordinary speech or TV | Wake word + command, threshold tuning, plenty of "unknown" data, measure per hour |
| `Harvard_TinyMLx` Arduino library is no longer actively updated | It is fine as the starting point; fall back to the upstream `tflite-micro` Arduino examples if it fails to build with a current core |
| Only one or two voices in our recordings | Split by speaker; collect from several people before judging accuracy |

## 6. References

- tinyMLx colabs: [3-5-18 Training KWS](https://github.com/tinyMLx/colabs/blob/master/3-5-18-TrainingKeywordSpotting.ipynb),
  [4-5-18 Favorite keywords](https://github.com/tinyMLx/colabs/blob/master/4-5-18-KWS-FavoriteKeywords.ipynb),
  [4-6-8 Custom dataset](https://github.com/tinyMLx/colabs/blob/master/4-6-8-CustomDatasetKWSModel.ipynb)
- Firmware reference: [tinyMLx micro_speech](https://github.com/tinyMLx/arduino-library/tree/main/examples/micro_speech)
- [Speech Commands on Hugging Face](https://huggingface.co/datasets/google/speech_commands)
- [MSWC on Hugging Face](https://huggingface.co/datasets/MLCommons/ml_spoken_words)
