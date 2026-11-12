"""Shared settings for training and firmware.

The audio settings must stay identical to the firmware's
`micro_features_micro_model_settings.h` (tinyMLx micro_speech example),
otherwise the model sees different features on the board than in training.
"""

from pathlib import Path

# --- Audio / feature settings (mirror of micro_model_settings.h) ---
SAMPLE_RATE = 16_000  # kAudioSampleFrequency
CLIP_DURATION_MS = 1_000
WINDOW_SIZE_MS = 30  # kFeatureSliceDurationMs
WINDOW_STRIDE_MS = 20  # kFeatureSliceStrideMs
FEATURE_BINS = 40  # kFeatureSliceSize
FEATURE_SLICES = 49  # kFeatureSliceCount
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION_MS // 1000

# --- Vocabulary ---
# Index 0 and 1 are fixed by the firmware (kSilenceIndex, kUnknownIndex).
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
# The wake word opens a short window in which the command words are accepted.
WAKE_WORD = "crystal"
KEYWORDS = [WAKE_WORD, "on", "off", "bright", "low"]
LABELS = [SILENCE_LABEL, UNKNOWN_LABEL, *KEYWORDS]

# Words fetched from MSWC. The first group are keywords or possible future
# keywords (all clips, up to the cap). The second group are words that sound
# like our keywords or are very common; they go into the "unknown" class so the
# model learns to tell "crystal" from "Christmas" and "on" from "one".
MSWC_KEYWORD_CANDIDATES = [
    *["crystal", "off", "bright", "low"],
    *["lamp", "star", "magic"],  # other wake word candidates
    *["dim", "light", "lights", "dark"],  # other command candidates
]
MSWC_KEYWORD_CAP = 4000
MSWC_UNKNOWN_WORDS = [
    *["crystals", "christmas", "christian", "christ", "chris", "cross", "crossed", "critical"],
    *["cricket", "crisis", "crucial", "crust", "crisp", "crest", "crush", "cruise", "criminal"],
    *["pistol", "bristol", "castle", "capital", "hospital", "coastal", "hostile", "whistle"],
    *["missile", "fiscal", "mystery", "history", "sister", "system", "kitchen", "script"],
    *["cluster", "custom", "start", "stars", "far", "magical"],
    *["lamb", "camp", "damp", "ramp", "lamps", "land", "lap", "lump", "lab", "plan", "lamport"],
    *["bride", "bite", "right", "write", "brought", "bread", "broad", "brown", "bring", "brights"],
    *["often", "offer", "office", "one", "own", "and", "done", "gone", "upon", "once", "under"],
    *["him", "them", "time", "team", "did", "been", "the", "that", "with", "have", "this"],
    *["from", "they", "what", "there", "were", "about", "would", "people", "because", "some"],
]
MSWC_UNKNOWN_CAP = 300

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
