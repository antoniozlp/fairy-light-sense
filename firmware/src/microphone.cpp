#include "microphone.h"

#include <Arduino.h>
#include <PDM.h>
#include <math.h>

#include "audio_provider.h"
#include "micro_features_micro_model_settings.h"

namespace {
int mic_gain = MIC_GAIN;
// GetAudioSamples() hands out at most kMaxAudioSampleSize samples per call.
constexpr int kChunkMs = kMaxAudioSampleSize / (kAudioSampleFrequency / 1000);
// The capture buffer holds 512 ms; stay well inside it.
constexpr int kLongestSpanMs = 400;
}  // namespace

void StartMicrophone(tflite::ErrorReporter* error_reporter, int gain) {
  // The first request for samples starts the capture, at the example's gain.
  int size = 0;
  int16_t* samples = nullptr;
  GetAudioSamples(error_reporter, 0, 0, &size, &samples);
  SetMicGain(gain);
}

int SetMicGain(int gain) {
  mic_gain = constrain(gain, kMicGainMin, kMicGainMax);
  PDM.setGain(mic_gain);
  return mic_gain;
}

int MicGain() { return mic_gain; }

float MicGainDb() { return (mic_gain - 40) / 2.0f; }

float MicPeakDb(tflite::ErrorReporter* error_reporter, int32_t from_ms,
                int32_t to_ms) {
  if (to_ms - from_ms > kLongestSpanMs) {
    from_ms = to_ms - kLongestSpanMs;
  }
  int lowest = 32767;
  int highest = -32768;
  for (int32_t start = from_ms; start < to_ms; start += kChunkMs) {
    const int duration = min<int32_t>(kChunkMs, to_ms - start);
    int size = 0;
    int16_t* samples = nullptr;
    if (GetAudioSamples(error_reporter, start, duration, &size, &samples) !=
        kTfLiteOk) {
      break;
    }
    const int count = duration * (kAudioSampleFrequency / 1000);
    for (int i = 0; i < count; ++i) {
      lowest = min<int>(lowest, samples[i]);
      highest = max<int>(highest, samples[i]);
    }
  }
  // Half the swing, so a constant offset of the microphone does not count.
  const float peak = max(1.0f, (highest - lowest) / 2.0f);
  return 20.0f * log10f(peak / 32768.0f);
}
