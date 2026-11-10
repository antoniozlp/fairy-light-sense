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
KEYWORDS = ["lamp", "on", "off", "bright", "dim"]
LABELS = [SILENCE_LABEL, UNKNOWN_LABEL, *KEYWORDS]

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
