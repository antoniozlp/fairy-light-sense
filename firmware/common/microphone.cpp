#include "microphone.h"

#include <math.h>

#include "audio_provider.h"
#include "micro_model_settings.h"

namespace {
int mic_gain = MIC_GAIN;
// GetAudioSamples() hands out at most kMaxAudioSampleSize samples per call.
constexpr int kChunkMs = kMaxAudioSampleSize / (kAudioSampleFrequency / 1000);
// The smaller capture buffer (the Nano's) holds 512 ms; stay well inside it.
constexpr int kLongestSpanMs = 400;
}  // namespace

bool StartMicrophone(int gain) {
  // The first request for samples starts the capture.
  int size = 0;
  int16_t* samples = nullptr;
  if (!GetAudioSamples(0, 0, &size, &samples)) {
    return false;
  }
  SetMicGain(gain);
  return true;
}

int SetMicGain(int gain) {
  mic_gain = gain < kMicGainMin ? kMicGainMin
                                : gain > kMicGainMax ? kMicGainMax : gain;
  ApplyMicGain(mic_gain);
  return mic_gain;
}

int MicGain() { return mic_gain; }

float MicGainDb() { return (mic_gain - 40) / 2.0f; }

float MicPeakDb(int32_t from_ms, int32_t to_ms) {
  if (to_ms - from_ms > kLongestSpanMs) {
    from_ms = to_ms - kLongestSpanMs;
  }
  int lowest = 32767;
  int highest = -32768;
  for (int32_t start = from_ms; start < to_ms; start += kChunkMs) {
    const int duration = to_ms - start < kChunkMs ? to_ms - start : kChunkMs;
    int size = 0;
    int16_t* samples = nullptr;
    if (!GetAudioSamples(start, duration, &size, &samples)) {
      break;
    }
    const int count = duration * (kAudioSampleFrequency / 1000);
    for (int i = 0; i < count; ++i) {
      if (samples[i] < lowest) lowest = samples[i];
      if (samples[i] > highest) highest = samples[i];
    }
  }
  // Half the swing, so a constant offset of the microphone does not count.
  const float peak = fmaxf(1.0f, (highest - lowest) / 2.0f);
  return 20.0f * log10f(peak / 32768.0f);
}
