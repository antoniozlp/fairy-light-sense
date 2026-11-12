"""Checks of the step that puts a trained model into the firmware sources."""

import numpy as np
import pytest

from fairy_kws import config, export, firmware, models


def exported_model(tmp_path, feature_max):
    rng = np.random.default_rng(0)
    features = rng.uniform(0, feature_max, (20, *models.INPUT_SHAPE)).astype(np.float32)
    model_dir = tmp_path / "crystal" / "tiny_conv"
    model_dir.mkdir(parents=True)
    tflite_model = export.to_int8_tflite(models.tiny_conv(len(config.LABELS)), features)
    (model_dir / "model_int8.tflite").write_bytes(tflite_model)
    (model_dir / "labels.txt").write_text("\n".join(config.LABELS) + "\n")
    return model_dir, tflite_model


def test_model_and_labels_are_written_together(tmp_path):
    model_dir, tflite_model = exported_model(tmp_path, feature_max=26)
    header, source = firmware.write_model_sources(model_dir, tmp_path / "firmware")

    assert header.parent == tmp_path / "firmware" / "models" / "crystal_tiny_conv"
    assert f"constexpr int kCategoryCount = {len(config.LABELS)};" in header.read_text()
    assert 'kModelName[] = "crystal_tiny_conv"' in header.read_text()
    text = source.read_text()
    assert f"g_model_len = {len(tflite_model)};" in text
    labels = [f'"{label}"' for label in config.LABELS]
    assert [text.index(label) for label in labels] == sorted(text.index(label) for label in labels)


def test_model_with_other_input_scaling_is_refused(tmp_path):
    # Features in 0..1 instead of 0..26: the board would feed this model wrong values.
    model_dir, _ = exported_model(tmp_path, feature_max=1)
    with pytest.raises(ValueError, match="input quantisation"):
        firmware.write_model_sources(model_dir, tmp_path / "firmware")


def test_environment_is_added_once(tmp_path):
    ini = tmp_path / "platformio.ini"
    ini.write_text("[env]\nboard = nano33ble\n\n[env:crystal_ds_cnn]\n")

    assert firmware.environment_name(tmp_path / "crystal_v1" / "ds_cnn") == "crystal_v1_ds_cnn"
    assert firmware.add_environment("crystal_v1_ds_cnn", ini)
    assert not firmware.add_environment("crystal_v1_ds_cnn", ini)
    assert not firmware.add_environment("crystal_ds_cnn", ini)
    assert ini.read_text().count("[env:crystal_v1_ds_cnn]") == 1
