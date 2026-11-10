# micro_speech

Audio capture and feature generation from the `micro_speech` example of
[tinyMLx/arduino-library](https://github.com/tinyMLx/arduino-library/tree/main/examples/micro_speech)
(TensorFlow Lite for Microcontrollers, Apache 2.0). The files are unchanged
except `micro_features_micro_model_settings.h`, where the yes/no category list
was removed: our categories are generated with the model into `src/model_data.h`.

These settings must stay identical to `src/fairy_kws/config.py`, otherwise the
model sees different features on the board than in training.
