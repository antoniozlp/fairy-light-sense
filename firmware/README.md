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

## What the board does

| Signal | Meaning |
|---|---|
| Blue LED on | the wake word was heard; a command is accepted for 2 s |
| Orange LED off / faint / dim / full | lamp OFF / LOW / ON / BRIGHT |
| Red LED blinking | set-up failed; the serial port repeats the reason |

The serial port (9600 baud) prints every detection and, every two seconds, a
status line with the classification rate, the inference time and the current
scores:

```
fairy-light-sense, model crystal/ds_cnn, 38232 bytes, tensor arena 30432 of 49152 bytes used
Say "crystal", then a command.
status: 3.1 classifications/s, inference 290.4 ms | crystal 0.00 on 0.00 off 0.00 bright 0.00 low 0.00 | lamp OFF
37.6 s  off 0.98  ->  ignored, say "crystal" first
```

## Changing the model

`src/model_data.h` and `src/model_data.cpp` are generated from a model folder of
the training notebook and hold the model together with its labels. From the
repository root:

```bash
uv run python -m fairy_kws.firmware models/crystal/tiny_conv
```

Without an argument it takes `models/crystal/ds_cnn`. The command refuses a
model whose input scaling differs from what the firmware computes. Both
architectures run without other changes: `main.cpp` registers the operations of
both and the tensor arena is sized for the larger one.

## Layout

```
├── platformio.ini
├── src/
│   ├── main.cpp               set-up and the listen/classify loop
│   ├── lamp_recognizer.*      averaging, threshold and the wake word rule
│   ├── lamp_output.*          LEDs (later: the LED string)
│   └── model_data.*           generated: model and labels
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
