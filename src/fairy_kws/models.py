"""Model architectures small enough for a microcontroller."""

import keras
from keras import layers

from fairy_kws import config

INPUT_SHAPE = (config.FEATURE_SLICES, config.FEATURE_BINS, 1)


def tiny_conv(num_labels: int, dropout: float = 0.5) -> keras.Model:
    """The `tiny_conv` model of the micro_speech example: one convolution, one dense layer."""
    return keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            layers.Conv2D(8, (10, 8), strides=(2, 2), padding="same", activation="relu"),
            layers.Dropout(dropout),
            layers.Flatten(),
            layers.Dense(num_labels, activation="softmax"),
        ],
        name="tiny_conv",
    )


def ds_cnn(num_labels: int, filters: int = 48, blocks: int = 4, dropout: float = 0.2):
    """Depthwise-separable CNN, after the keyword spotting reference model of MLPerf Tiny.

    The first convolution shrinks the 49 x 40 input to 25 x 10; the blocks that
    follow keep that size. About ten times the computation of `tiny_conv`
    (roughly 3.7 million multiply-adds per inference against 0.34 million).
    """

    def conv_bn_relu(x, layer):
        x = layer(x)
        # momentum 0.9: the running statistics settle within the first epoch
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.ReLU()(x)

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = conv_bn_relu(
        inputs, layers.Conv2D(filters, (10, 8), strides=(2, 4), padding="same", use_bias=False)
    )
    x = layers.Dropout(dropout)(x)
    for _ in range(blocks):
        x = conv_bn_relu(x, layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False))
        x = conv_bn_relu(x, layers.Conv2D(filters, (1, 1), use_bias=False))
    x = layers.Dropout(dropout)(x)
    x = layers.AveragePooling2D(pool_size=x.shape[1:3])(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_labels, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="ds_cnn")


ARCHITECTURES = {"tiny_conv": tiny_conv, "ds_cnn": ds_cnn}
