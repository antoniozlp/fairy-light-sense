# models/

Training outputs (Keras checkpoints, `.tflite` files, generated C arrays) are
written here and are not committed. The model that ships in the firmware is
copied into `firmware/` when a version is ready to flash.

`02_train_keywords.ipynb` writes to `models/<wake word>/<architecture>/`, so
models trained for different wake words sit side by side:

```
models/
├── crystal/{tiny_conv,ds_cnn}/   # current wake word
└── lamp/{tiny_conv,ds_cnn}/      # first wake word, kept for comparison
```
