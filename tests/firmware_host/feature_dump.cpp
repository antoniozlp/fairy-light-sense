// Runs the firmware's feature code on the computer.
//
// Reads one second of 16-bit, 16 kHz audio from the file given as argument and
// writes the 49 x 40 int8 values the board would hand to the model. In place
// of a microphone, GetAudioSamples() serves the samples of the file.
// tests/test_feature_parity.py compares the output with the training features.

#include <stdio.h>

#include <vector>

#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_model_settings.h"

namespace {
std::vector<int16_t> clip;
int16_t output_buffer[kMaxAudioSampleSize];
}  // namespace

bool GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  const size_t start = static_cast<size_t>(start_ms) * (kAudioSampleFrequency / 1000);
  const int count = duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < count; ++i) {
    output_buffer[i] = start + i < clip.size() ? clip[start + i] : 0;
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = output_buffer;
  return true;
}

int32_t LatestAudioTimestamp() { return 0; }
void ApplyMicGain(int) {}

int main(int argc, char** argv) {
  FILE* file = argc == 2 ? fopen(argv[1], "rb") : nullptr;
  if (file == nullptr) {
    fprintf(stderr, "usage: feature_dump <raw int16 audio file>\n");
    return 2;
  }
  int16_t sample;
  while (fread(&sample, sizeof(sample), 1, file) == 1) {
    clip.push_back(sample);
  }
  fclose(file);

  static int8_t features[kFeatureElementCount];
  FeatureProvider provider(kFeatureElementCount, features);
  // At this time the 49 slices start at 0, 20, ..., 960 ms: the whole clip.
  const int32_t now_ms = (kFeatureSliceCount - 1) * kFeatureSliceStrideMs;
  int new_slices = 0;
  if (!provider.PopulateFeatureData(0, now_ms, &new_slices) ||
      new_slices != kFeatureSliceCount) {
    fprintf(stderr, "feature generation failed\n");
    return 1;
  }
  fwrite(features, 1, sizeof(features), stdout);
  return 0;
}
