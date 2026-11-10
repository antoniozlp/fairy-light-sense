"""Audio -> model input features, using the same front end as the firmware.

TensorFlow Lite Micro's micro_speech example computes its features on the
board with the "micro frontend" (filterbank + noise reduction + PCAN + log
scale). TensorFlow ships the same code as a Python op, so training on its
output keeps the board and the notebook in agreement.
"""

import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import (
    audio_microfrontend_op as frontend_op,
)

from fairy_kws import config

# The firmware turns each front end value v (0..~666) into an int8 with
# ((v * 256) + 333) / 666 - 128. Scaling by 10/256 here gives floats in
# 0..~26, which the int8 input quantization maps onto that same range.
FEATURE_SCALE = 10.0 / 256.0


def pad_or_trim(audio: tf.Tensor, length: int = config.CLIP_SAMPLES) -> tf.Tensor:
    """Force a 1-D waveform to exactly `length` samples."""
    audio = audio[:length]
    return tf.pad(audio, [[0, length - tf.shape(audio)[0]]])


def micro_features(audio: tf.Tensor) -> tf.Tensor:
    """Convert one clip to a (FEATURE_SLICES, FEATURE_BINS) float32 feature map.

    `audio` is a 1-D waveform at 16 kHz, either float in [-1, 1] or int16.
    """
    if audio.dtype != tf.int16:
        audio = tf.cast(tf.clip_by_value(audio, -1.0, 1.0) * 32767.0, tf.int16)
    audio = pad_or_trim(audio)
    features = frontend_op.audio_microfrontend(
        audio,
        sample_rate=config.SAMPLE_RATE,
        window_size=config.WINDOW_SIZE_MS,
        window_step=config.WINDOW_STRIDE_MS,
        num_channels=config.FEATURE_BINS,
        out_scale=1,
        out_type=tf.float32,
    )
    return features * FEATURE_SCALE
