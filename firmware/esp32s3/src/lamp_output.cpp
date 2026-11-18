// ESP32-S3-DevKitC-1: the board has one addressable RGB LED. It shows the lamp
// state as warm white of rising brightness, turns blue while a command is
// awaited, and red for a failed set-up.

#include "lamp_output.h"

#include "board.h"
#include "led_strip.h"

namespace {

struct Color {
  uint8_t red, green, blue;
};

constexpr Color kOff = {0, 0, 0};
constexpr Color kLow = {3, 2, 0};
constexpr Color kOn = {40, 24, 6};
constexpr Color kBright = {255, 150, 40};
constexpr Color kListening = {0, 0, 40};
constexpr Color kFailure = {60, 0, 0};

// One LED on each of the two possible pins (see board.h).
constexpr int kPins[] = {RGB_LED_PIN, RGB_LED_PIN_V1_0};
constexpr int kPinCount = sizeof(kPins) / sizeof(kPins[0]);
led_strip_handle_t leds[kPinCount] = {};

Color lamp_color = kOff;
bool is_listening = false;

void Show(Color color) {
  for (led_strip_handle_t led : leds) {
    if (led != nullptr) {
      led_strip_set_pixel(led, 0, color.red, color.green, color.blue);
      led_strip_refresh(led);
    }
  }
}

}  // namespace

void SetupLampOutput() {
  for (int i = 0; i < kPinCount; ++i) {
    if (kPins[i] < 0) {
      continue;
    }
    led_strip_config_t strip_config = {};
    strip_config.strip_gpio_num = kPins[i];
    strip_config.max_leds = 1;
    strip_config.led_model = LED_MODEL_WS2812;
    strip_config.color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB;
    led_strip_rmt_config_t rmt_config = {};
    rmt_config.resolution_hz = 10 * 1000 * 1000;
    // One LED needs 24 symbols; 48 is the smallest block of the ESP32-S3.
    rmt_config.mem_block_symbols = 48;
    if (led_strip_new_rmt_device(&strip_config, &rmt_config, &leds[i]) !=
        ESP_OK) {
      leds[i] = nullptr;
    }
  }
  SetLamp(LampState::kOff);
}

void SetLamp(LampState state) {
  switch (state) {
    case LampState::kOff:
      lamp_color = kOff;
      break;
    case LampState::kLow:
      lamp_color = kLow;
      break;
    case LampState::kOn:
      lamp_color = kOn;
      break;
    case LampState::kBright:
      lamp_color = kBright;
      break;
  }
  Show(is_listening ? kListening : lamp_color);
}

void ShowListening(bool listening) {
  // Called on every loop; the LED is only written when the state changes.
  if (listening != is_listening) {
    is_listening = listening;
    Show(is_listening ? kListening : lamp_color);
  }
}

void ShowFailure(bool on) { Show(on ? kFailure : kOff); }
