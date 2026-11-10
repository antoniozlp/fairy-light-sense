"""Find the audio clips, split them, and turn them into model inputs.

This is a TensorFlow 2 version of what `speech_commands/input_data.py` does in
the tinyMLx colabs: keyword clips plus an "unknown" class made of other words
and a "silence" class made of background noise, with time shift and background
noise added to training clips.
"""

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from fairy_kws import config, mswc
from fairy_kws.features import micro_features

GSC_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GSC_DIR = config.DATA_DIR / "speech_commands_v0.02"
GSC_ARCHIVE = config.DATA_DIR / "downloads" / "speech_commands_v0.02.tar.gz"
BACKGROUND_DIR = GSC_DIR / "_background_noise_"

# Augmentation settings. Time shift and noise volume are the values of the tinyMLx
# colabs. Unlike there, background noise is added to every training clip, not 80% of
# them: MSWC clips are short word cuts padded to one second with digital silence,
# which a real microphone never produces.
TIME_SHIFT_MS = 100
BACKGROUND_VOLUME = 0.1


def download_speech_commands() -> Path:
    """Download and unpack Google Speech Commands v2 (2.4 GB) unless already there."""
    if not BACKGROUND_DIR.exists():
        mswc.download(GSC_URL, GSC_ARCHIVE)
        GSC_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(GSC_ARCHIVE) as tar:
            tar.extractall(GSC_DIR, filter="data")
    return GSC_DIR


# --------------------------------------------------------------------------
# Clip table
# --------------------------------------------------------------------------


def _speech_commands_clips() -> pd.DataFrame:
    """Speech Commands clips with the dataset's own train/val/test split (by speaker)."""

    def listed(name):
        return set((GSC_DIR / name).read_text().split())

    val, test = listed("validation_list.txt"), listed("testing_list.txt")
    rows = []
    for path in GSC_DIR.glob("*/*.wav"):
        word = path.parent.name
        if word.startswith("_"):
            continue
        key = f"{word}/{path.name}"
        split = "val" if key in val else "test" if key in test else "train"
        rows.append((str(path), word, "speech_commands", split))
    return pd.DataFrame(rows, columns=["path", "word", "source", "split"])


def _mswc_clips() -> pd.DataFrame:
    """MSWC clips on disk with MSWC's own split, which also keeps speakers apart."""
    if not mswc.SPLITS_PATH.exists():
        return pd.DataFrame(columns=["path", "word", "source", "split"])
    splits = pd.read_csv(mswc.SPLITS_PATH)
    splits["path"] = [str((mswc.CLIPS_DIR / link).with_suffix(".wav")) for link in splits["LINK"]]
    splits["split"] = splits["SET"].map({"TRAIN": "train", "DEV": "val", "TEST": "test"})
    splits = splits.rename(columns={"WORD": "word"}).assign(source="mswc")
    on_disk = {str(p) for p in mswc.CLIPS_DIR.glob("*/*.wav")}
    return splits.loc[splits["path"].isin(on_disk), ["path", "word", "source", "split"]]


def build_clip_table(keywords: list[str] = config.KEYWORDS) -> pd.DataFrame:
    """One row per clip: path, word, source, split and the label the model should learn.

    Words that are not keywords get the label "_unknown_".
    """
    clips = pd.concat([_speech_commands_clips(), _mswc_clips()], ignore_index=True)
    clips["label"] = clips["word"].where(clips["word"].isin(keywords), config.UNKNOWN_LABEL)
    return clips.sort_values("path", ignore_index=True)


# --------------------------------------------------------------------------
# Audio loading and augmentation
# --------------------------------------------------------------------------


def load_clip(path: tf.Tensor) -> tf.Tensor:
    """Read a WAV file as float32 in [-1, 1], padded or cut to exactly one second."""
    audio, _ = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1, desired_samples=config.CLIP_SAMPLES
    )
    return tf.squeeze(audio, axis=-1)


def load_background_noise() -> tf.Tensor:
    """All Speech Commands background recordings joined into one long waveform."""
    clips = []
    for path in sorted(BACKGROUND_DIR.glob("*.wav")):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(str(path)), desired_channels=1)
        clips.append(tf.squeeze(audio, axis=-1))
    return tf.concat(clips, axis=0)


def random_background(noise: tf.Tensor, max_volume: float = BACKGROUND_VOLUME) -> tf.Tensor:
    """One second of background noise from a random position, at a random volume."""
    start = tf.random.uniform([], 0, tf.shape(noise)[0] - config.CLIP_SAMPLES, dtype=tf.int32)
    volume = tf.random.uniform([], 0.0, max_volume)
    return noise[start : start + config.CLIP_SAMPLES] * volume


def augment(audio: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
    """Shift the clip up to 100 ms either way, then add background noise."""
    max_shift = config.SAMPLE_RATE * TIME_SHIFT_MS // 1000
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    padded = tf.pad(audio, [[max_shift, max_shift]])
    audio = padded[max_shift - shift : max_shift - shift + config.CLIP_SAMPLES]
    return tf.clip_by_value(audio + random_background(noise), -1.0, 1.0)


def _to_model_input(audio: tf.Tensor) -> tf.Tensor:
    features = micro_features(audio)[..., tf.newaxis]
    return tf.ensure_shape(features, (config.FEATURE_SLICES, config.FEATURE_BINS, 1))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def training_dataset(
    clips: pd.DataFrame,
    labels: list[str] = config.LABELS,
    class_weights: dict[str, float] | None = None,
    batch_size: int = 100,
    seed: int = 0,
) -> tf.data.Dataset:
    """Endless stream of augmented training batches with balanced classes.

    Every label has its own shuffled, repeating stream of clips and each example
    is drawn from one of them at random, so a word with few clips
    appears as often as a word with many. `class_weights` changes those odds.

    Within a label, the two datasets are drawn from equally. Otherwise the
    unknown class would be 85% Speech Commands words, and the model would see
    few of the MSWC words chosen because they sound like our keywords.
    """

    def clip_stream(paths, stream_seed):
        return (
            tf.data.Dataset.from_tensor_slices(paths)
            .shuffle(len(paths), seed=stream_seed)
            .repeat()
            .map(load_clip, num_parallel_calls=tf.data.AUTOTUNE)
        )

    class_weights = class_weights or {}
    train = clips[clips["split"] == "train"]
    noise = load_background_noise()
    streams, weights = [], []
    for index, label in enumerate(labels):
        if label == config.SILENCE_LABEL:
            stream = tf.data.Dataset.from_tensors(tf.zeros(config.CLIP_SAMPLES)).repeat()
        else:
            by_source = [
                clip_stream(group["path"].to_numpy(), seed + 100 * index + i)
                for i, (_, group) in enumerate(train[train["label"] == label].groupby("source"))
            ]
            if not by_source:
                raise ValueError(f"no training clips for {label!r}")
            stream = tf.data.Dataset.sample_from_datasets(by_source, seed=seed + index)
        streams.append(stream.map(lambda audio, i=index: (audio, i)))
        weights.append(class_weights.get(label, 1.0))
    weights = [w / sum(weights) for w in weights]
    return (
        tf.data.Dataset.sample_from_datasets(streams, weights, seed=seed)
        .map(
            lambda audio, label: (_to_model_input(augment(audio, noise)), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def evaluation_set(
    clips: pd.DataFrame,
    split: str,
    labels: list[str] = config.LABELS,
    background: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fixed evaluation data for `split` ("val" or "test").

    Contains every keyword clip of the split, as many unknown clips as keyword
    clips in total, and as many silence clips (background noise only) as the
    average keyword has. With `background`, each clip is mixed with background
    noise at a random volume, as in training but with a fixed random seed, so
    the set is identical on every call. Clips are never time-shifted.

    Returns features, label indices, and the table of the clips used (silence
    rows have no path).
    """
    rng = np.random.default_rng(seed)
    part = clips[clips["split"] == split]
    keywords = part[part["label"] != config.UNKNOWN_LABEL]
    unknown = part[part["label"] == config.UNKNOWN_LABEL]
    unknown = unknown.sample(min(len(unknown), len(keywords)), random_state=seed)
    table = pd.concat([keywords, unknown], ignore_index=True)
    n_silence = len(keywords) // max(1, keywords["label"].nunique())

    noise = load_background_noise()
    n = len(table) + n_silence
    starts = rng.integers(0, len(noise) - config.CLIP_SAMPLES, n).astype(np.int32)
    volumes = rng.uniform(0.0, BACKGROUND_VOLUME, n).astype(np.float32)
    if not background:
        volumes[: len(table)] = 0.0  # silence clips always consist of noise

    speech = tf.data.Dataset.from_tensor_slices(table["path"].to_numpy()).map(
        load_clip, num_parallel_calls=tf.data.AUTOTUNE
    )
    silence = tf.data.Dataset.from_tensors(tf.zeros(config.CLIP_SAMPLES)).repeat(n_silence)

    def add_noise(audio, start, volume):
        mixed = audio + noise[start : start + config.CLIP_SAMPLES] * volume
        return tf.clip_by_value(mixed, -1.0, 1.0)

    features = np.concatenate(
        list(
            tf.data.Dataset.zip(
                speech.concatenate(silence), tf.data.Dataset.from_tensor_slices((starts, volumes))
            )
            .map(lambda audio, mix: _to_model_input(add_noise(audio, *mix)))
            .batch(256)
            .as_numpy_iterator()
        )
    )
    silence_rows = pd.DataFrame(
        {"word": config.SILENCE_LABEL, "label": config.SILENCE_LABEL, "source": "noise"},
        index=range(n_silence),
    ).assign(split=split)
    table = pd.concat([table, silence_rows], ignore_index=True)
    y = table["label"].map({label: i for i, label in enumerate(labels)}).to_numpy()
    return features, y, table
