# Project plan: voice-controlled fairy light lamp

Goal: a fairy light lamp that responds to spoken commands ("lamp on", "lamp off",
"lamp bright", …) recognised on the microcontroller itself, with no phone or cloud.
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

The reference colabs classify single one-second words. "Lamp on" is two words, and
there are two ways to handle that:

| | A. Single words + state machine (recommended) | B. Whole phrase as one class |
|---|---|---|
| Model classes | silence, unknown, `lamp`, `on`, `off`, `bright`, `dim` | silence, unknown, `lamp on`, `lamp off`, `lamp bright`, … |
| Audio window | 1 s, 49×40 features (unchanged from `micro_speech`) | 1.5–2 s, about twice the input and compute |
| Public data | `on`/`off` exist in Speech Commands; `lamp`/`bright`/`dim` partly in MSWC | none; every clip must be recorded or synthesised |
| Firmware | hear `lamp`, then accept a command word for about 1.5 s | direct mapping, no state machine |
| False triggers | low: two detections in a row are needed | depends entirely on the custom data |
| Adding a command | one new word | one new phrase, new recordings |

Recommendation: **start with A**. It reuses the proven feature pipeline and public
data, and `lamp` acts as a wake word, which matters because short words like "on"
are easily confused with "one" or "off" in normal conversation. B stays open as a
later experiment once there is a recording workflow.

### 1.3 Train in TensorFlow 2, not with the TF 1.15 scripts

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
| [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands) | `on`, `off`; 33 other words for the "unknown" class; background noise clips | CC BY 4.0 | About 2.4 GB, roughly 105,000 one-second clips, 35 words. Does **not** contain `lamp`, `bright`, `dim` or `light`. |
| [Multilingual Spoken Words Corpus (MSWC)](https://mlcommons.org/datasets/multilingual-spoken-words/) | Candidate source for `lamp`, `bright`, `dim`, `light`, `off` | CC BY 4.0 | One-second clips cut from Common Voice. English audio is a 32 GB archive; a 103 MB metadata file lists every word with its clip count. **To verify:** how many clips exist for each of our words. Very short words such as "on" are likely absent. |
| Own recordings | All keywords, in the voices and rooms that matter | ours | The tinyMLx custom-dataset colab uses the Open Speech Recording web tool. Recording through the board's own microphone gives the best match to deployment. Target: 50–100 clips per word per speaker, several speakers. |
| Synthetic speech (TTS) | Hundreds of voices for any word, cheaply | depends on the voice model | The approach used by the [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) project (Piper TTS plus noise and room augmentation). Good for volume; real recordings are still needed for validation. |

Plan: Speech Commands for `on`/`off`, unknown and noise; MSWC for `lamp`,
`bright`, `dim` if the clip counts are sufficient (a few hundred each at minimum);
own recordings to fill gaps and to form a test set that reflects real use; TTS only
if a word is still short of data.

Candidate vocabulary: `lamp` (wake word), `on`, `off`, `bright`, `dim`. "Dim" is
suggested as the counterpart of "bright"; "light" is a possible alternative to
"lamp" but is a far more common word in conversation.

## 3. Model

| Model | Size (int8) | Role |
|---|---|---|
| `tiny_conv` (one conv layer + dense, from `micro_speech`) | 31 kB with our 7 classes (measured in the environment check notebook) | Baseline. Runs unchanged in the reference firmware. |
| DS-CNN small (MLPerf Tiny keyword spotting reference) | roughly 50 kB (published figure, not yet measured here) | Step up if `tiny_conv` accuracy is not enough for 7 classes. |

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

### Phase 1: Data
- `01_data_exploration.ipynb`: download Speech Commands, listen, plot features.
- Check MSWC clip counts for the candidate words; extract them if sufficient.
- Record own clips; script to trim to 1 s and normalise (16 kHz mono WAV).
- Build one `tf.data` pipeline: train/validation/test split by speaker, background
  noise mixing, time shift, silence and unknown classes.

### Phase 2: Baseline model on public data only
- `02_train_baseline.ipynb`: `tiny_conv` on `on`/`off` + unknown + silence.
- Goal: reproduce the accuracy of the tinyMLx colab in TensorFlow 2. This proves
  the training pipeline before custom words add uncertainty.
- Convert to int8, compare float and quantised accuracy, export the C array.

### Phase 3: Baseline on the board
- Install Arduino IDE or `arduino-cli`, the Nano 33 BLE core and the
  `Harvard_TinyMLx` library.
- Flash stock `micro_speech`, then swap in our model, labels and category count.
- **Feature parity test**: print the features the board computes and compare with
  the notebook's features for similar audio. This is the most common place for a
  model to work in the notebook and fail on the device.
- `on`/`off` drive the built-in LED.

### Phase 4: Full vocabulary
- `03_train_lamp_keywords.ipynb`: add `lamp`, `bright`, `dim`.
- Evaluate on the recorded test set and on long negative audio (false activations
  per hour); tune the detection threshold and averaging window.
- Try DS-CNN if needed.

### Phase 5: Lamp behaviour and hardware
- Firmware state machine: `lamp` opens a ~1.5 s window; `on`/`off`/`bright`/`dim`
  act only inside it.
- Drive the LED string through a logic-level MOSFET with PWM (the board's pins are
  3.3 V and cannot power the string directly); brightness steps for bright/dim.
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
| Too little data for `lamp`/`bright`/`dim` | Check MSWC counts early; record own data; TTS as a supplement |
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
