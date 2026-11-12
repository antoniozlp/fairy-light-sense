"""Checks of the live listening pipeline that need no microphone and no trained model."""

import os

import numpy as np
import soundfile as sf

from fairy_kws import config, export, listen, models

LABELS = config.LABELS
WAKE, COMMAND = LABELS[2], LABELS[3]


def scores_for(word):
    scores = np.full(len(LABELS), 0.01, dtype=np.float32)
    scores[LABELS.index(word)] = 0.95
    return scores


def feed(recognizer, word, start, seconds=0.5, hop=0.1):
    """Feed the same scores for a while; return the detections and the end time."""
    detections = []
    for step in range(1, round(seconds / hop) + 1):
        detection = recognizer.update(scores_for(word), start + step * hop)
        if detection:
            detections.append(detection)
    return detections, start + seconds


def test_wake_word_then_command_is_accepted_once():
    recognizer = listen.Recognizer(LABELS)
    wake, now = feed(recognizer, WAKE, 0.0)
    assert [(d.word, d.accepted) for d in wake] == [(WAKE, True)]  # reported once, not per hop
    assert recognizer.listening(now)
    command, now = feed(recognizer, COMMAND, now)
    assert [(d.word, d.accepted) for d in command] == [(COMMAND, True)]
    assert not recognizer.listening(now)  # one command per wake word


def test_command_without_wake_word_is_ignored():
    recognizer = listen.Recognizer(LABELS)
    detections, _ = feed(recognizer, COMMAND, 0.0)
    assert [(d.word, d.accepted) for d in detections] == [(COMMAND, False)]


def test_command_after_the_window_is_ignored():
    recognizer = listen.Recognizer(LABELS, command_window_s=2.0)
    _, now = feed(recognizer, WAKE, 0.0)
    _, now = feed(recognizer, config.SILENCE_LABEL, now, seconds=3.0)
    detections, _ = feed(recognizer, COMMAND, now)
    assert [(d.word, d.accepted) for d in detections] == [(COMMAND, False)]


def test_low_scores_and_unknown_words_are_not_detections():
    recognizer = listen.Recognizer(LABELS, threshold=0.7)
    weak = np.full(len(LABELS), 0.1, dtype=np.float32)
    weak[LABELS.index(WAKE)] = 0.5
    assert all(recognizer.update(weak, 0.1 * step) is None for step in range(1, 20))
    detections, _ = feed(recognizer, config.UNKNOWN_LABEL, 2.0)
    assert detections == []


def test_status_line_fits_the_terminal(monkeypatch):
    class Stub:
        labels = LABELS

    recognizer = listen.Recognizer(LABELS)
    block = np.full(1600, 0.1, dtype=np.float32)
    for columns in (40, 80, 200):
        size = os.terminal_size((columns, 24))
        monkeypatch.setattr(listen.shutil, "get_terminal_size", lambda size=size: size)
        line = listen._status_line(Stub, recognizer, block, 0.0, "bright")
        assert len(line) < columns  # a wrapped line could not be overwritten in place
    assert line.startswith("waiting") and "lamp BRIGHT" in line
    assert line.endswith(f"{LABELS[-1]} 0.00")  # nothing is cut off at 200 columns


def test_sound_file_runs_through_the_whole_pipeline(tmp_path, capsys):
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 26, (20, *models.INPUT_SHAPE)).astype(np.float32)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tflite_model = export.to_int8_tflite(models.tiny_conv(len(LABELS)), features)
    (model_dir / "model_int8.tflite").write_bytes(tflite_model)
    (model_dir / "labels.txt").write_text("\n".join(LABELS) + "\n")
    # two seconds at 48 kHz, so the file also goes through the conversion to 16 kHz
    sf.write(tmp_path / "noise.wav", rng.uniform(-0.01, 0.01, 96_000), 48_000)

    blocks = list(listen.file_blocks(tmp_path / "noise.wav", hop_s=0.1))
    assert len(blocks) == 20 and all(len(block) == 1600 for block in blocks)
    scores = listen.Classifier(model_dir).push(blocks[0])
    assert scores.shape == (len(LABELS),)
    np.testing.assert_allclose(scores.sum(), 1.0, atol=0.02)

    # The model folder is accepted with and without --model.
    for model_args in ([str(model_dir)], ["--model", str(model_dir)]):
        listen.main([*model_args, "--file", str(tmp_path / "noise.wav")])
        assert f'Say "{WAKE}"' in capsys.readouterr().out
