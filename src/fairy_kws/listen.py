"""Try a trained model live on the computer's microphone.

Does what the firmware will do on the board: about ten times per second it
takes the last second of audio, computes the features, runs the int8 model, and
passes the scores to a recogniser that averages them and applies the lamp's
rule (the wake word first, then a command within a short time).

    uv run python -m fairy_kws.listen                       # default model and microphone
    uv run python -m fairy_kws.listen --list-devices
    uv run python -m fairy_kws.listen --model models/crystal/tiny_conv --threshold 0.9
    uv run python -m fairy_kws.listen --file recording.wav  # a WAV file instead of the microphone

Stop with Ctrl+C.
"""

import argparse
import queue
import shutil
import sys
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly

from fairy_kws import config
from fairy_kws.export import Int8Model
from fairy_kws.features import micro_features

FIRST_KEYWORD = 2  # labels 0 and 1 are silence and unknown


@dataclass
class Detection:
    """A keyword the recogniser heard. `accepted` is False for a command without the wake word."""

    time: float
    word: str
    score: float
    accepted: bool


class Recognizer:
    """Turns the stream of model scores into detections and applies the wake word rule.

    Scores are averaged over the last `average_s` seconds; a keyword is detected
    when its average is the highest and at least `threshold`. The same word is
    not reported again for `suppression_s` seconds, because one spoken word
    stays inside the one-second window for many classifications in a row. The
    wake word (the first keyword of the model) opens a window of
    `command_window_s` seconds in which one command is accepted.

    The defaults did best when test clips of the wake word and a command were
    joined into simulated streams; they are a starting point for real voices.
    """

    def __init__(
        self,
        labels: list[str],
        threshold: float = 0.7,
        average_s: float = 0.3,
        suppression_s: float = 1.5,
        command_window_s: float = 2.0,
        min_count: int = 2,
    ):
        self.labels = labels
        self.wake_word = labels[FIRST_KEYWORD]
        self.threshold = threshold
        self.average_s = average_s
        self.suppression_s = suppression_s
        self.command_window_s = command_window_s
        self.min_count = min_count
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self.average = np.zeros(len(labels), dtype=np.float32)
        self.last_heard: dict[str, float] = {}
        self.listening_until = float("-inf")

    def listening(self, now: float) -> bool:
        """True while a command would be accepted."""
        return now <= self.listening_until

    def update(self, scores: np.ndarray, now: float) -> Detection | None:
        """Add the scores of one classification made at time `now` (seconds)."""
        self.history.append((now, scores))
        while self.history[0][0] <= now - self.average_s:
            self.history.popleft()
        self.average = np.mean([s for _, s in self.history], axis=0)
        top = int(self.average.argmax())
        word, score = self.labels[top], float(self.average[top])
        if len(self.history) < self.min_count or top < FIRST_KEYWORD or score < self.threshold:
            return None
        if now - self.last_heard.get(word, float("-inf")) < self.suppression_s:
            return None
        self.last_heard[word] = now
        if word == self.wake_word:
            self.listening_until = now + self.command_window_s
            return Detection(now, word, score, accepted=True)
        accepted = self.listening(now)
        self.listening_until = float("-inf")
        return Detection(now, word, score, accepted)


class Classifier:
    """Keeps the last second of audio and classifies it with an exported model."""

    def __init__(self, model_dir: Path):
        self.labels = (model_dir / "labels.txt").read_text().split()
        self.model = Int8Model((model_dir / "model_int8.tflite").read_bytes())
        self.audio = np.zeros(config.CLIP_SAMPLES, dtype=np.float32)
        signature = [tf.TensorSpec([config.CLIP_SAMPLES], tf.float32)]
        self.features = tf.function(micro_features, input_signature=signature)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Append new 16 kHz samples and return the scores for the last second."""
        self.audio = np.concatenate([self.audio, samples])[-config.CLIP_SAMPLES :]
        features = self.features(tf.constant(self.audio)).numpy()
        return self.model.predict(features[np.newaxis, ..., np.newaxis])[0]


# --------------------------------------------------------------------------
# Audio sources: both yield mono float32 blocks at 16 kHz
# --------------------------------------------------------------------------


def _to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == config.SAMPLE_RATE:
        return samples
    divisor = gcd(rate, config.SAMPLE_RATE)
    resampled = resample_poly(samples, config.SAMPLE_RATE // divisor, rate // divisor)
    return resampled.astype(np.float32)


def file_blocks(path: Path, hop_s: float) -> Iterator[np.ndarray]:
    """The audio of a sound file, cut into blocks of `hop_s` seconds."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = _to_16k(audio.mean(axis=1), rate)
    hop = round(hop_s * config.SAMPLE_RATE)
    for start in range(0, len(audio) - hop + 1, hop):
        yield audio[start : start + hop]


def microphone_blocks(device: int | str | None, hop_s: float) -> Iterator[np.ndarray]:
    """Blocks of `hop_s` seconds from a microphone, until interrupted."""
    import sounddevice as sd  # imported here so that --file works without an audio device

    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time, status):
        blocks.put(indata[:, 0].copy())

    def stream(rate):
        return sd.InputStream(
            device=device,
            channels=1,
            samplerate=rate,
            blocksize=round(hop_s * rate),
            dtype="float32",
            callback=callback,
        )

    # Ask for 16 kHz. If the device refuses, record at its own rate and convert.
    rate = config.SAMPLE_RATE
    try:
        sd.check_input_settings(device=device, channels=1, samplerate=rate, dtype="float32")
    except sd.PortAudioError:
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
    warned, count, silent = False, 0, True
    with stream(rate):
        while True:
            if blocks.qsize() > 20 and not warned:
                print("\nThe computer cannot keep up with the microphone; try a larger --hop.")
                warned = True
            block = blocks.get()
            count += 1
            silent = silent and not block.any()
            if silent and count == round(2 / hop_s):
                # macOS hands a program without microphone permission pure zeros
                print(
                    "\nThe microphone delivers only silence. On macOS, allow your terminal under"
                    " System Settings > Privacy & Security > Microphone, then start again."
                )
            yield _to_16k(block, rate)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _status_line(classifier, recognizer, block, now, lamp) -> str:
    level = 20 * np.log10(max(float(np.sqrt(np.mean(block**2))), 1e-5))  # dBFS, -100..0
    meter = "#" * int(np.clip((level + 60) / 7.5, 0, 8))
    scores = " ".join(
        f"{label} {score:.2f}"
        for label, score in zip(classifier.labels, recognizer.average, strict=True)
        if not label.startswith("_")
    )
    state = "LISTENING" if recognizer.listening(now) else "waiting"
    line = f"{state:<9} lamp {lamp.upper():<6} [{meter:<8}] {scores}"
    # A line wider than the terminal wraps, and then it can no longer be overwritten in place.
    return line[: shutil.get_terminal_size().columns - 1]


def run(
    blocks: Iterator[np.ndarray],
    classifier: Classifier,
    recognizer: Recognizer,
    hop_s: float,
    live: bool,
) -> list[Detection]:
    """Classify the blocks as they arrive and print what is heard. Returns all detections."""
    detections, lamp, now = [], "off", 0.0
    try:
        for block in blocks:
            now += hop_s
            detection = recognizer.update(classifier.push(block), now)
            if detection:
                detections.append(detection)
                if detection.word == recognizer.wake_word:
                    result = f"listening for a command for {recognizer.command_window_s:g} s"
                elif detection.accepted:
                    lamp = detection.word
                    result = f"lamp {lamp.upper()}"
                else:
                    result = f'ignored, say "{recognizer.wake_word}" first'
                event = f"{now:7.1f} s  {detection.word:<8} {detection.score:.2f}  ->  {result}"
                print(f"\r\033[K{event}" if live else event)
            if live:
                status = _status_line(classifier, recognizer, block, now, lamp)
                print(f"\r\033[K{status}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    if live:
        print()
    return detections


def main(argv: list[str] | None = None) -> list[Detection]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add = parser.add_argument
    add(
        "--model",
        type=Path,
        default=config.MODELS_DIR / config.WAKE_WORD / "ds_cnn",
        help="folder written by the training notebook (default: %(default)s)",
    )
    add("--threshold", type=float, default=0.7, help="score a keyword needs (default: 0.7)")
    add("--window", type=float, default=2.0, help="seconds a command is accepted (default: 2)")
    add("--average", type=float, default=0.3, help="seconds of scores averaged (default: 0.3)")
    add("--hop", type=float, default=0.1, help="seconds between classifications (default: 0.1)")
    add("--device", help="microphone name or number, see --list-devices")
    add("--list-devices", action="store_true", help="show the microphones and exit")
    add("--file", type=Path, help="classify a sound file instead of the microphone")
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd

        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                default = "  (default)" if index == sd.default.device[0] else ""
                print(f"{index}: {device['name']}{default}")
        return []

    if not (args.model / "model_int8.tflite").exists():
        parser.error(f"no model in {args.model}; run notebooks/02_train_keywords.ipynb first")
    classifier = Classifier(args.model)
    recognizer = Recognizer(
        classifier.labels,
        threshold=args.threshold,
        average_s=args.average,
        command_window_s=args.window,
        # with a long hop, fewer classifications fall inside the averaging time
        min_count=min(2, max(1, int(args.average / args.hop))),
    )
    commands = ", ".join(classifier.labels[FIRST_KEYWORD + 1 :])
    print(f"model: {args.model}")
    print(f'Say "{recognizer.wake_word}", then one of: {commands}. Ctrl+C to stop.')

    if args.file:
        blocks = file_blocks(args.file, args.hop)
    else:
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        blocks = microphone_blocks(device, args.hop)
    detections = run(blocks, classifier, recognizer, args.hop, live=sys.stdout.isatty())

    counts = Counter((d.word, d.accepted) for d in detections)
    for (word, accepted), count in sorted(counts.items()):
        print(f"{word}: {count}" + ("" if accepted else " (ignored, no wake word)"))
    return detections


if __name__ == "__main__":
    main()
