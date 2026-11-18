# Firmware for the Arduino Nano 33 BLE Sense

PlatformIO project (Arduino core) for the Nano 33 BLE Sense with its on-board
PDM microphone. The code it shares with the ESP32-S3 is in `../common`, the
models in `../models`; see [../README.md](../README.md) for both.

## Build and flash

Requires [PlatformIO Core](https://docs.platformio.org/en/latest/core/index.html)
(`pio`). Run these from this folder with the board connected over USB.

```bash
pio run --target upload
```

```bash
pio device monitor
```

`pio run` without `-e` builds the default model, `crystal_ds_cnn`; another one
is flashed with `-e <model>`. The first build downloads the `Harvard_TinyMLx`
library (TensorFlow Lite for Microcontrollers 2.4 with CMSIS-NN).

Close any serial monitor (`pio device monitor`, `tio`, ...) before flashing. With
a monitor open the upload either cannot open the port or loses the board half
way, which leaves it waiting in its bootloader (orange LED fading in and out).
If that happens, close the monitor and upload again.

## What the board shows

| Signal | Meaning |
|---|---|
| Blue LED on | the wake word was heard; a command is accepted for 2 s |
| Orange LED off / faint / dim / full | lamp OFF / LOW / ON / BRIGHT |
| Red LED blinking | set-up failed; the serial port repeats the reason |

## Microphone gain

The gain is the setting of the nRF52840's PDM input. The board starts at 20
(-10 dB), the value of the micro_speech example (`kDefaultMicGain` in
`src/board.h`).

## Files

```
├── platformio.ini       one environment per model
└── src/
    ├── board.h          board name, tensor arena size, default gain
    ├── audio_provider.cpp   PDM capture (from the micro_speech example)
    └── lamp_output.cpp  the orange, blue and red LEDs
```
