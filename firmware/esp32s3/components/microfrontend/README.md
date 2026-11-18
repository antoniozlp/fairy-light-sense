# microfrontend

The audio front end of TensorFlow Lite for Microcontrollers
(`tensorflow/lite/experimental/microfrontend/lib`) with its copy of KISS FFT,
taken from
[tinyMLx/arduino-library](https://github.com/tinyMLx/arduino-library)
(Apache 2.0; KISS FFT is BSD, see `third_party/kissfft/COPYING`). One line was
added: `kiss_fft.c` now includes `string.h`, which current compilers require.

It turns audio into the 40 values per 20 ms that the model reads. The Nano 33
BLE Sense build gets this code from the `Harvard_TinyMLx` library. Espressif's
TensorFlow Lite Micro component no longer ships it, so the ESP32-S3 build
compiles this copy: the same code as on the Nano, and so the same features as
in training.
