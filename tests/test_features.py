import numpy as np
import tensorflow as tf

from fairy_kws import config
from fairy_kws.features import micro_features


def _tone(seconds: float = 1.0, freq: float = 440.0) -> tf.Tensor:
    t = np.arange(int(config.SAMPLE_RATE * seconds)) / config.SAMPLE_RATE
    return tf.constant(0.5 * np.sin(2 * np.pi * freq * t), dtype=tf.float32)


def test_feature_shape_matches_firmware():
    features = micro_features(_tone())
    assert features.shape == (config.FEATURE_SLICES, config.FEATURE_BINS)


def test_feature_range_fits_int8_mapping():
    features = micro_features(_tone()).numpy()
    assert features.min() >= 0.0
    assert features.max() <= 26.1  # 666 * 10 / 256


def test_short_and_long_clips_are_padded_or_trimmed():
    assert micro_features(_tone(0.4)).shape == micro_features(_tone(1.7)).shape


def test_int16_and_float_input_agree():
    tone = _tone()
    as_int16 = tf.cast(tone * 32767.0, tf.int16)
    np.testing.assert_allclose(micro_features(tone), micro_features(as_int16))


def test_labels_keep_firmware_indices():
    assert config.LABELS[0] == config.SILENCE_LABEL
    assert config.LABELS[1] == config.UNKNOWN_LABEL
