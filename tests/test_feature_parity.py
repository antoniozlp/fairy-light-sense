"""The firmware's feature code, compiled for the computer, against the training features.

Both boards build the feature code in firmware/common with the audio front end
in firmware/esp32s3/components/microfrontend (the Nano gets the same front end
from its TensorFlow Lite library). If its output differed from `micro_features`, the
model would see something on the board that it never saw in training.
"""

import shutil
import subprocess

import numpy as np
import pytest

from fairy_kws import config
from fairy_kws.features import micro_features
from fairy_kws.firmware import FIRMWARE_INPUT_SCALE, FIRMWARE_INPUT_ZERO_POINT

FIRMWARE = config.REPO_ROOT / "firmware"
HOST = config.REPO_ROOT / "tests" / "firmware_host"
COMMON = FIRMWARE / "common"
FRONTEND = FIRMWARE / "esp32s3" / "components" / "microfrontend"
FRONTEND_LIB = FRONTEND / "tensorflow" / "lite" / "experimental" / "microfrontend" / "lib"
SOURCES = [
    HOST / "feature_dump.cpp",
    COMMON / "feature_provider.cpp",
    COMMON / "micro_features_generator.cpp",
    *sorted(FRONTEND_LIB.glob("*.c")),
    *sorted(FRONTEND_LIB.glob("*.cpp")),
    FRONTEND / "third_party" / "kissfft" / "kiss_fft.c",
    FRONTEND / "third_party" / "kissfft" / "tools" / "kiss_fftr.c",
]


@pytest.fixture(scope="module")
def feature_dump(tmp_path_factory):
    """Path of the compiled program; skips the tests without a C++ compiler."""
    if not (shutil.which("cc") and shutil.which("c++")):
        pytest.skip("needs a C and C++ compiler")
    build = tmp_path_factory.mktemp("feature_dump")
    includes = [f"-I{COMMON}", f"-I{FRONTEND}"]
    objects = []
    for source in SOURCES:
        compiler = ["cc"] if source.suffix == ".c" else ["c++", "-std=c++17"]
        objects.append(build / f"{source.stem}.o")
        subprocess.run(
            [*compiler, "-O1", "-w", *includes, "-c", str(source), "-o", str(objects[-1])],
            check=True,
        )
    program = build / "feature_dump"
    subprocess.run(["c++", *map(str, objects), "-lm", "-o", str(program)], check=True)
    return program


def clips():
    rng = np.random.default_rng(0)
    t = np.arange(config.CLIP_SAMPLES) / config.SAMPLE_RATE
    chirp = 0.5 * np.sin(2 * np.pi * (200 + 1500 * t) * t)
    burst = np.where((t > 0.3) & (t < 0.6), rng.uniform(-0.8, 0.8, len(t)), 0.0)
    return {
        "noise": rng.uniform(-0.05, 0.05, len(t)),
        "chirp": chirp,
        "burst in quiet noise": burst + rng.uniform(-0.01, 0.01, len(t)),
    }


@pytest.mark.parametrize("name", list(clips()))
def test_firmware_features_match_training_features(name, feature_dump, tmp_path):
    audio = (clips()[name] * 32767).astype(np.int16)
    (tmp_path / "clip.raw").write_bytes(audio.astype("<i2").tobytes())
    out = subprocess.run(
        [str(feature_dump), str(tmp_path / "clip.raw")], check=True, capture_output=True
    ).stdout
    board = np.frombuffer(out, dtype=np.int8).reshape(config.FEATURE_SLICES, config.FEATURE_BINS)

    # What the int8 model input receives for the training features.
    trained = micro_features(audio).numpy() / FIRMWARE_INPUT_SCALE + FIRMWARE_INPUT_ZERO_POINT
    trained = np.clip(np.round(trained), -128, 127)

    assert board.std() > 5  # not a flat picture
    # The board divides by 666 where the exact factor is 665.6: one step at most.
    assert np.abs(board.astype(int) - trained).max() <= 1
