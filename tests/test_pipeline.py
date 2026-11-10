"""Checks that need no downloaded data."""

import io

import numpy as np
import pytest
import soundfile as sf
import tensorflow as tf

from fairy_kws import config, dataset, export, models, mswc


@pytest.mark.parametrize("name", list(models.ARCHITECTURES))
def test_models_output_one_probability_per_label(name):
    model = models.ARCHITECTURES[name](len(config.LABELS))
    scores = model(np.zeros((2, *models.INPUT_SHAPE), dtype=np.float32)).numpy()
    assert scores.shape == (2, len(config.LABELS))
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-5)


def test_int8_conversion_keeps_predictions_and_uses_static_shapes(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 26, (40, *models.INPUT_SHAPE)).astype(np.float32)
    model = models.tiny_conv(len(config.LABELS))
    tflite_model = export.to_int8_tflite(model, features)

    scores = export.tflite_predict(tflite_model, features)
    agreement = (scores.argmax(axis=1) == model.predict(features, verbose=0).argmax(axis=1)).mean()
    assert agreement >= 0.9
    # Shape-computing operations would mean the batch size was left flexible.
    assert set(export.tflite_ops(tflite_model)) <= {
        "CONV_2D",
        "FULLY_CONNECTED",
        "RESHAPE",
        "SOFTMAX",
    }

    source = export.write_c_source(tflite_model, tmp_path / "model.cc").read_text()
    assert f"g_model_len = {len(tflite_model)};" in source


def test_augment_keeps_length_and_range():
    audio = tf.constant(np.linspace(-1, 1, config.CLIP_SAMPLES), dtype=tf.float32)
    noise = tf.random.uniform([5 * config.CLIP_SAMPLES], -1.0, 1.0, seed=0)
    out = dataset.augment(audio, noise).numpy()
    assert out.shape == (config.CLIP_SAMPLES,)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_mswc_clip_is_resampled_to_one_second_at_16k(tmp_path):
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(48_000 * 3 // 4) / 48_000)
    encoded = io.BytesIO()
    sf.write(encoded, tone, 48_000, format="OGG", subtype="OPUS")
    mswc.opus_to_wav(encoded.getvalue(), tmp_path / "word" / "clip.wav")

    audio, rate = sf.read(tmp_path / "word" / "clip.wav", dtype="int16")
    assert rate == config.SAMPLE_RATE
    assert audio.shape == (config.CLIP_SAMPLES,)
    assert np.abs(audio[: config.SAMPLE_RATE // 2]).max() > 5000  # the tone
    assert np.abs(audio[-1000:]).max() == 0  # padding after 0.75 s


def test_wanted_clips_applies_caps():
    index = {"lamp": [f"{i}.opus" for i in range(10)], "lamb": [f"{i}.opus" for i in range(999)]}
    wanted = mswc.wanted_clips(index)
    assert len(wanted["lamp"]) == 10
    assert len(wanted["lamb"]) == config.MSWC_UNKNOWN_CAP
