# Firmware

The lamp's firmware for two boards, one PlatformIO project each. Both listen
with a microphone, run the int8 keyword model with TensorFlow Lite for
Microcontrollers, and apply the lamp's rule: say "crystal", then "light",
"off", "bright" or "low" within two seconds.

| | [Arduino Nano 33 BLE Sense](nano33ble/README.md) | [ESP32-S3-DevKitC-1](esp32s3/README.md) |
|---|---|---|
| Project folder | `nano33ble/` | `esp32s3/` |
| Processor | nRF52840, Cortex-M4F, 64 MHz | ESP32-S3, two Xtensa LX7 cores, 240 MHz |
| Microphone | on the board, PDM | INMP441 breakout, I2S |
| Framework | Arduino | ESP-IDF |
| TensorFlow Lite Micro | 2.4 from `Harvard_TinyMLx`, CMSIS-NN kernels | Espressif's `esp-tflite-micro`, ESP-NN kernels |
| Serial port | 9600 baud | 115200 baud, USB port labelled "UART" |

Each board's README says how to build and flash it and what its LEDs show; the
ESP32-S3's also says how to wire the microphone.

```
firmware/
├── nano33ble/                 PlatformIO project of the Nano 33 BLE Sense
├── esp32s3/                   PlatformIO project of the ESP32-S3-DevKitC-1
├── common/                    code both projects build
│   ├── main.cpp               set-up and the listen/classify loop
│   ├── lamp_recognizer.*      averaging, threshold and the wake word rule
│   ├── feature_provider.*     keeps the 49 x 40 feature map of the last second
│   ├── micro_features_generator.*  audio to features, with the audio front end
│   ├── microphone.*           gain setting and level meter
│   ├── audio_provider.h       what a board's audio capture has to offer
│   └── lamp_output.*          what a board's LEDs have to offer, state names
└── models/<name>/model_data.* generated: a model and its labels, used by both
```

What a board has to provide (audio capture, LEDs and its `board.h` with the
serial port and a few settings) is in its own project's `src/`. The rest is
the same code on both, so the two boards cannot drift apart.
`lamp_recognizer` follows `Recognizer` in `src/fairy_kws/listen.py`, so the
computer test and the boards apply the same rule with the same settings. The
feature code comes from the micro_speech example of
[tinyMLx/arduino-library](https://github.com/tinyMLx/arduino-library); its
settings (`micro_model_settings.h`) must stay identical to
`src/fairy_kws/config.py`. `tests/test_feature_parity.py` compiles it on the
computer and checks it against the training features.

## Models

Every model is an environment of the same name in both projects, built with
the model in `models/<name>/`. Flash one by name from a board's folder:

```bash
cd firmware/nano33ble && pio run -e crystal_v1_ds_cnn --target upload
```

```bash
cd firmware/esp32s3 && pio run -e crystal_v1_ds_cnn --target upload
```

| Environment | Model | Commands |
|---|---|---|
| `crystal_ds_cnn` (default) | `models/crystal/ds_cnn` | light, off, bright, low |
| `crystal_v2_ds_cnn` | `models/crystal_v2/ds_cnn` | on, off, bright, low |
| `crystal_v1_ds_cnn` | `models/crystal_v1/ds_cnn` | on, off, bright |

"light" and "on" both switch the lamp to ON; a model has one of the two.

Each environment keeps its own build, so after the first build of each,
switching takes only the upload. Every status line on the serial port starts
with the name of the model that is running.

To add a model, or to refresh one after training, run this from the repository
root with a model folder of the training notebook:

```bash
uv run python -m fairy_kws.firmware models/crystal/tiny_conv
```

It writes `models/crystal_tiny_conv/model_data.h` and `model_data.cpp` (the
model and its labels) and adds `[env:crystal_tiny_conv]` to the
`platformio.ini` of both boards. The name comes from the last two folder
names; `--name` sets another. Without an argument it takes
`models/crystal/ds_cnn`. The command refuses a model whose input scaling
differs from what the firmware computes. Both architectures run without other
changes: `main.cpp` registers the operations of both and each board's tensor
arena is sized for the larger one.

## Serial output

Both boards print every detection and, every two seconds, a status line with
the classification rate, the inference time, the current scores and the
microphone level:

```
fairy-light-sense on nano33ble, model crystal_ds_cnn, 38232 bytes, tensor arena 30432 of 49152 bytes used
mic gain 20 (-10.0 dB), "+" and "-" change it
Say "crystal", then a command.
[crystal_ds_cnn] 3.1/s 290 ms | crystal 0.00 light 0.00 off 0.00 bright 0.00 low 0.00 | mic -52 dB gain 20 | lamp OFF
37.6 s  off 0.98  mic -14 dB  ->  ignored, say "crystal" first
```

## Microphone gain

`mic` is the loudest level the microphone picked up, in dB below full scale:
0 dB is clipping. A status line shows the loudest level since the previous
one, a detection the loudest of about the last second. The clips the models
were trained on peak at about -6 to -12 dB, so a voice that arrives far below
that is harder to recognise.

The gain goes from 0 to 80 in steps of half a decibel; 40 is neutral (0 dB).
Each board starts at its own default (see its README). To change it:

- **While the board runs:** type `+` or `-` in the serial monitor. Each key
  press is 3 dB and the board prints the new gain. The change lasts until the
  board restarts.
- **For good:** set `kDefaultMicGain` in the board's `src/board.h`, or add
  `-D MIC_GAIN=<0..80>` to `build_flags` in the board's `platformio.ini`.

To find a value, speak from where the board should still hear you and raise the
gain until your voice peaks near -10 dB; then speak close to the board and
check that it stays below 0 dB. More gain raises the room noise by the same
amount.

## Comparing the two boards

Both boards print the same status line, so the numbers can be read side by
side. Flash the same model on both and put them next to each other.

1. **Speed.** The start of each status line, for example `3.1/s 290 ms`, is
   the number of classifications per second and the time of one inference.
2. **Microphone level.** Speak from the same place and compare `mic` on the
   detection lines. Bring the two within a few dB of each other with `+` and
   `-`; otherwise one board simply hears you louder.
3. **Recognition.** Say each command the same number of times from the same
   place (for example ten times "crystal", pause, "light") and count the
   lines that end in `lamp ...` on each board. Repeat from further away.
4. **False triggers.** Leave both next to a radio or TV for the same time and
   count the detections.

The models and the audio features are identical on both boards. They differ
in the microphone, in the TensorFlow Lite Micro version and its kernels (so
scores can differ slightly), and in speed: a faster board classifies more
often, which gives the recogniser more scores to average over each word.

## Measured on the boards

`ds_cnn`, model `crystal_ds_cnn`:

| | Nano 33 BLE Sense | ESP32-S3-DevKitC-1 |
|---|---|---|
| Inference | 290 ms | not measured yet |
| Classifications per second | 3.1 | not measured yet |
| Tensor arena used | 30.4 kB | not measured yet |
| Flash / RAM of the whole firmware | 221 kB / 114 kB | 424 kB / 150 kB (96 kB of it reserved for the tensor arena) |

The ESP32-S3 firmware builds but has not run on a board yet.
