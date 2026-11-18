# Firmware for the ESP32-S3-DevKitC-1

PlatformIO project (ESP-IDF) for the ESP32-S3-DevKitC-1 with an INMP441
microphone on I2S. The code it shares with the Nano 33 BLE Sense is in
`../common`, the models in `../models`; see [../README.md](../README.md) for
both.

## Wiring the INMP441

The INMP441 is an I2S microphone: the ESP32-S3 sends it two clocks and reads
the sound as numbers on one data line. Six wires, all on header J1 (the side
with the 3V3 pins) except that any ground pin will do:

| INMP441 | ESP32-S3-DevKitC-1 | Purpose |
|---|---|---|
| VDD | 3V3 (J1 pin 1) | power: 3.3 V, never 5 V |
| GND | G (J1 pin 22) | ground |
| L/R | G (any ground pin) | selects the left channel; do not leave it open |
| SCK | GPIO4 (J1 pin 4) | bit clock, 1.024 MHz |
| WS | GPIO5 (J1 pin 5) | word select, 16 kHz: one sample per period |
| SD | GPIO6 (J1 pin 6) | data from the microphone |

- Keep the wires short (10 cm or less): the bit clock is fast.
- The sound hole is on the side of the breakout without components. Point it
  towards the room.
- Other pins: set `MIC_PIN_SCK`, `MIC_PIN_WS`, `MIC_PIN_SD` as build flags
  (`-D MIC_PIN_SD=7`), see `src/board.h`. Avoid GPIO0, 3, 45 and 46
  (boot mode), 19 and 20 (USB), 43 and 44 (serial port), and 35 to 37 on
  boards with 8 MB of PSRAM or more.
- With L/R at 3V3 the microphone sends on the right channel: build with
  `-D MIC_RIGHT_CHANNEL=1`.
- If the `mic` level on the serial port stays at -90 dB, no data arrives:
  check SD, the two clocks and L/R.

## Build and flash

Requires [PlatformIO Core](https://docs.platformio.org/en/latest/core/index.html)
(`pio`). Connect the board through its USB port labelled "UART" and run these
from this folder:

```bash
pio run --target upload
```

```bash
pio device monitor
```

`pio run` without `-e` builds the default model, `crystal_ds_cnn`; another one
is flashed with `-e <model>`. The first build downloads ESP-IDF, its compilers
and two components (`esp-tflite-micro`, `led_strip`); that takes several
minutes and about 4.5 GB of disk in `~/.platformio`. Close any serial monitor
before flashing.

## What the board shows

The board has one RGB LED, on GPIO38 (board revision v1.1) or GPIO48 (v1.0).
The firmware drives both pins, so either revision lights up.

| Colour | Meaning |
|---|---|
| Blue | the wake word was heard; a command is accepted for 2 s |
| Off / faint / dim / full warm white | lamp OFF / LOW / ON / BRIGHT |
| Red, blinking | set-up failed; the serial port repeats the reason |

## Microphone gain

The INMP441 has no gain setting, so the firmware multiplies its samples: 40
keeps the upper 16 of the microphone's 24 bits. The board starts at 64
(+12 dB), the level Espressif's micro_speech example uses
(`kDefaultMicGain` in `src/board.h`).

## Files

```
├── platformio.ini       one environment per model, board settings
├── CMakeLists.txt       ESP-IDF project
├── sdkconfig.defaults   240 MHz, optimised for speed, 8 MB flash, 1 ms ticks
├── dependencies.lock    versions of the fetched components
├── components/microfrontend/   the audio front end (the Nano's library has its own copy)
└── src/
    ├── CMakeLists.txt   sources: these files, ../../common and the model
    ├── idf_component.yml    components to fetch: esp-tflite-micro, led_strip
    ├── board.h, board.cpp   pins, settings, and the few Arduino calls main.cpp uses
    ├── audio_provider.cpp   I2S capture of the INMP441
    └── lamp_output.cpp  the RGB LED
```

This build has no Arduino core. `src/board.*` provides the calls of
`../common/main.cpp` that come from Arduino (`Serial.print`, `millis`,
`delay`) on top of ESP-IDF, and starts `setup()` and `loop()`.
