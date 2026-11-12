# Firmware

PlatformIO project for the Arduino Nano 33 BLE Sense. It listens with the
on-board microphone, runs the int8 keyword model with TensorFlow Lite for
Microcontrollers, and applies the lamp's rule: say "crystal", then "on", "off",
"bright" or "low" within two seconds.

## Build and flash

Requires [PlatformIO Core](https://docs.platformio.org/en/latest/core/index.html)
(`pio`). Run these from this folder with the board connected over USB.

```bash
pio run --target upload
```

```bash
pio device monitor
```

The first build downloads the `Harvard_TinyMLx` library (TensorFlow Lite for
Microcontrollers 2.4 with CMSIS-NN).

Close any serial monitor (`pio device monitor`, `tio`, ...) before flashing. With
a monitor open the upload either cannot open the port or loses the board half
way, which leaves it waiting in its bootloader (orange LED fading in and out).
If that happens, close the monitor and upload again.

## Models

Every model is a PlatformIO environment: the same sources in `src/`, built with
the model in `models/<name>/`. Flash one by name:

```bash
pio run -e crystal_v1_ds_cnn --target upload
```

| Environment | Model | Commands |
|---|---|---|
| `crystal_ds_cnn` (default) | `models/crystal/ds_cnn` | on, off, bright, low |
| `crystal_v1_ds_cnn` | `models/crystal_v1/ds_cnn` | on, off, bright |

Each environment keeps its own build, so after the first build of each,
switching takes only the upload (about 15 seconds). Every status line on the
serial port starts with the name of the model that is running.

To add a model, or to refresh one after training, run this from the repository
root with a model folder of the training notebook:

```bash
uv run python -m fairy_kws.firmware models/crystal/tiny_conv
```

It writes `models/crystal_tiny_conv/model_data.h` and `model_data.cpp` (the
model and its labels) and adds `[env:crystal_tiny_conv]` to `platformio.ini`.
The name comes from the last two folder names; `--name` sets another. Without
an argument it takes `models/crystal/ds_cnn`. The command refuses a model whose
input scaling differs from what the firmware computes. Both architectures run
without other changes: `main.cpp` registers the operations of both and the
tensor arena is sized for the larger one.

## What the board does

| Signal | Meaning |
|---|---|
| Blue LED on | the wake word was heard; a command is accepted for 2 s |
| Orange LED off / faint / dim / full | lamp OFF / LOW / ON / BRIGHT |
| Red LED blinking | set-up failed; the serial port repeats the reason |

The serial port (9600 baud) prints every detection and, every two seconds, a
status line with the classification rate, the inference time, the current
scores and the microphone level:

```
fairy-light-sense, model crystal_ds_cnn, 38232 bytes, tensor arena 30432 of 49152 bytes used
mic gain 20 (-10.0 dB), "+" and "-" change it
Say "crystal", then a command.
[crystal_ds_cnn] 3.1/s 290 ms | crystal 0.00 on 0.00 off 0.00 bright 0.00 low 0.00 | mic -52 dB gain 20 | lamp OFF
37.6 s  off 0.98  mic -14 dB  ->  ignored, say "crystal" first
```

## Microphone gain

`mic` is the loudest level the microphone picked up, in dB below full scale:
0 dB is clipping. A status line shows the loudest level since the previous
one, a detection the loudest of about the last second. The clips the models
were trained on peak at about -6 to -12 dB, so a voice that arrives far below
that is harder to recognise.

The gain goes from 0 to 80 in steps of half a decibel; 40 is neutral (0 dB).
The board starts at 20 (-10 dB), the value of the micro_speech example. To
change it:

- **While the board runs:** type `+` or `-` in the serial monitor. Each key
  press is 3 dB and the board prints the new gain. The change lasts until the
  board restarts.
- **For good:** set `MIC_GAIN` in `src/microphone.h`, or add
  `-D MIC_GAIN=<0..80>` to `build_flags` in `platformio.ini`.

To find a value, speak from where the board should still hear you and raise the
gain until your voice peaks near -10 dB; then speak close to the board and
check that it stays below 0 dB. More gain raises the room noise by the same
amount.

## Layout

```
├── platformio.ini             one environment per model
├── src/
│   ├── main.cpp               set-up and the listen/classify loop
│   ├── lamp_recognizer.*      averaging, threshold and the wake word rule
│   ├── microphone.*           microphone gain and level
│   └── lamp_output.*          LEDs (later: the LED string)
├── models/<name>/model_data.* generated: a model and its labels
└── lib/micro_speech/          microphone and feature code of the micro_speech example
```

`lamp_recognizer` follows `Recognizer` in `src/fairy_kws/listen.py`, so the
computer test and the board apply the same rule with the same settings.

## Measured on the board

| | `ds_cnn` |
|---|---|
| Inference | 290 ms |
| Classifications per second | 3.1 |
| Tensor arena used | 30.4 kB |
| Flash / RAM of the whole sketch | 219 kB / 114 kB |
