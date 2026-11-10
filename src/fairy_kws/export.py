"""Turn a trained Keras model into what the microcontroller needs."""

import contextlib
import io
import warnings
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

# TensorFlow 2.21 warns that tf.lite.Interpreter will move to the separate LiteRT
# package. It still works in the version pinned by uv.lock, so keep notebooks readable.
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
warnings.filterwarnings("ignore", message="Statistics for quantized inputs were expected.*")


def to_int8_tflite(model: keras.Model, representative_features: np.ndarray) -> bytes:
    """Convert to a TensorFlow Lite model whose weights, inputs and outputs are all int8.

    `representative_features` are real model inputs; the converter runs them
    through the model to learn the value range of every layer.
    """

    def representative_dataset():
        for example in representative_features:
            yield [example[np.newaxis].astype(np.float32)]

    # Fix the batch size to 1. With a flexible batch size the converter adds shape
    # operations (SHAPE, PACK, STRIDED_SLICE) that the firmware would have to support.
    inputs = keras.Input(batch_shape=(1, *model.input_shape[1:]))
    single = keras.Model(inputs, model(inputs))
    converter = tf.lite.TFLiteConverter.from_keras_model(single)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # The converter prints a long export log; silence it for the duration of the call.
    logger = tf.get_logger()
    level = logger.level
    logger.setLevel("ERROR")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return converter.convert()
    finally:
        logger.setLevel(level)


class Int8Model:
    """An int8 TensorFlow Lite model kept loaded, for running it many times.

    Inputs are quantised and outputs de-quantised with the scales stored in the
    model, which is what the firmware does on the board.
    """

    def __init__(self, tflite_model: bytes):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for a batch of feature maps, one row per example."""
        in_scale, in_zero = self.input["quantization"]
        out_scale, out_zero = self.output["quantization"]
        quantized = np.clip(np.round(features / in_scale + in_zero), -128, 127).astype(np.int8)
        scores = np.empty((len(features), self.output["shape"][-1]), dtype=np.float32)
        for i, example in enumerate(quantized):
            self.interpreter.set_tensor(self.input["index"], example[np.newaxis])
            self.interpreter.invoke()
            raw = self.interpreter.get_tensor(self.output["index"])[0]
            scores[i] = (raw.astype(np.float32) - out_zero) * out_scale
        return scores


def tflite_predict(tflite_model: bytes, features: np.ndarray) -> np.ndarray:
    """Run an int8 TensorFlow Lite model, returning class probabilities per example."""
    return Int8Model(tflite_model).predict(features)


def tflite_ops(tflite_model: bytes) -> list[str]:
    """Names of the operations in the model. The firmware must register each of them."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    # No public API lists the operators, so this uses the interpreter's internal one.
    return sorted({op["op_name"] for op in interpreter._get_ops_details()})


def c_array(tflite_model: bytes, name: str = "g_model") -> str:
    """The model as C source: a byte array `name` and its length `name_len`."""
    rows = [
        "  " + ", ".join(f"0x{b:02x}" for b in tflite_model[i : i + 12])
        for i in range(0, len(tflite_model), 12)
    ]
    return (
        f"alignas(16) const unsigned char {name}[] = {{\n"
        + ",\n".join(rows)
        + f"\n}};\nconst int {name}_len = {len(tflite_model)};\n"
    )


def write_c_source(tflite_model: bytes, path: Path, name: str = "g_model") -> Path:
    """Write the model as a C array, like `xxd -i` does in the tinyMLx colabs."""
    path.write_text(c_array(tflite_model, name))
    return path
