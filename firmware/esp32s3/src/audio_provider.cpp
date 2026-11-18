// Audio capture for the ESP32-S3 with an INMP441 microphone.
//
// The Nano 33 BLE Sense has a PDM microphone that the nRF52840 converts to
// samples. The INMP441 is an I2S microphone: the ESP32-S3 supplies a bit clock
// (SCK) and a word select clock (WS), and the microphone answers on SD with
// one 24-bit sample per WS period, sent first in a 32-bit slot.
//
// A task reads those samples, scales them to 16 bits and stores them in a ring
// buffer indexed by sample number since start, like the Nano's capture code.
// The feature code then asks for "the samples from millisecond X", the same
// way on both boards.

#include "audio_provider.h"

#include <math.h>

#include "board.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "micro_model_settings.h"

namespace {

// One second of audio. The main loop reads the newest samples many times a
// second, long before they are overwritten.
constexpr int kCaptureBufferSize = kAudioSampleFrequency;
// The task reads 20 ms at a time, one feature stride.
constexpr int kChunkSamples = kAudioSampleFrequency / 50;

int16_t g_capture_buffer[kCaptureBufferSize];
int16_t g_output_buffer[kMaxAudioSampleSize];
int32_t g_i2s_samples[kChunkSamples];

bool g_is_audio_initialized = false;
i2s_chan_handle_t g_i2s_rx = nullptr;
// Milliseconds of audio captured so far.
volatile int32_t g_latest_audio_timestamp = 0;
// Factor from the 24-bit sample to the 16-bit value that is stored. At 0 dB it
// is 1/256: the upper 16 bits.
volatile float g_sample_scale = 1.0f / 256.0f;

void CaptureTask(void*) {
  int64_t sample_count = 0;
  while (true) {
    size_t bytes_read = 0;
    if (i2s_channel_read(g_i2s_rx, g_i2s_samples, sizeof(g_i2s_samples),
                         &bytes_read, portMAX_DELAY) != ESP_OK) {
      continue;
    }
    const int count = bytes_read / sizeof(int32_t);
    const float scale = g_sample_scale;
    for (int i = 0; i < count; ++i) {
      // The 24-bit sample sits in the upper bits of the 32-bit slot.
      float value = (g_i2s_samples[i] >> 8) * scale;
      if (value > 32767.0f) value = 32767.0f;
      if (value < -32768.0f) value = -32768.0f;
      g_capture_buffer[(sample_count + i) % kCaptureBufferSize] =
          static_cast<int16_t>(lrintf(value));
    }
    sample_count += count;
    // This is how the main loop knows that new audio has arrived.
    g_latest_audio_timestamp =
        static_cast<int32_t>(sample_count / (kAudioSampleFrequency / 1000));
  }
}

bool InitAudioRecording() {
  i2s_chan_config_t channel_config =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  if (i2s_new_channel(&channel_config, nullptr, &g_i2s_rx) != ESP_OK) {
    return false;
  }

  // Standard (Philips) I2S, 16 kHz, 32-bit slots: 64 clock periods per sample,
  // which the INMP441 requires. Only the slot of the microphone is kept.
  i2s_std_config_t config = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kAudioSampleFrequency),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                      I2S_SLOT_MODE_MONO),
      .gpio_cfg =
          {
              .mclk = I2S_GPIO_UNUSED,
              .bclk = static_cast<gpio_num_t>(MIC_PIN_SCK),
              .ws = static_cast<gpio_num_t>(MIC_PIN_WS),
              .dout = I2S_GPIO_UNUSED,
              .din = static_cast<gpio_num_t>(MIC_PIN_SD),
              .invert_flags =
                  {
                      .mclk_inv = false,
                      .bclk_inv = false,
                      .ws_inv = false,
                  },
          },
  };
  config.slot_cfg.slot_mask =
      MIC_RIGHT_CHANNEL ? I2S_STD_SLOT_RIGHT : I2S_STD_SLOT_LEFT;
  if (i2s_channel_init_std_mode(g_i2s_rx, &config) != ESP_OK ||
      i2s_channel_enable(g_i2s_rx) != ESP_OK) {
    return false;
  }

  if (xTaskCreate(CaptureTask, "audio_capture", 4096, nullptr, 10, nullptr) !=
      pdPASS) {
    return false;
  }
  // Block until the first samples have arrived.
  while (!g_latest_audio_timestamp) {
    vTaskDelay(1);
  }
  return true;
}

}  // namespace

bool GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    const bool initialized = InitAudioRecording();
    if (!initialized) {
      return false;
    }
    g_is_audio_initialized = true;
  }
  // Index, in the history of all samples, of the first sample wanted.
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  int duration_sample_count = duration_ms * (kAudioSampleFrequency / 1000);
  if (duration_sample_count > kMaxAudioSampleSize) {
    duration_sample_count = kMaxAudioSampleSize;
  }
  for (int i = 0; i < duration_sample_count; ++i) {
    g_output_buffer[i] = g_capture_buffer[(start_offset + i) % kCaptureBufferSize];
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_output_buffer;
  return true;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }

void ApplyMicGain(int gain) {
  // Half a decibel per step, 40 is 0 dB.
  g_sample_scale = powf(10.0f, (gain - 40) / 40.0f) / 256.0f;
}
