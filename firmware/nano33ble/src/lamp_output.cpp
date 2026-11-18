// Nano 33 BLE Sense: the orange built-in LED is the lamp, the blue LED shows
// that a command is awaited.

#include "lamp_output.h"

#include <Arduino.h>

namespace {
const int kLampPin = LED_BUILTIN;
// PWM duty (0..255) for each lamp state.
const int kDutyOff = 0;
const int kDutyLow = 6;
const int kDutyOn = 40;
const int kDutyBright = 255;
}  // namespace

void SetupLampOutput() {
  pinMode(kLampPin, OUTPUT);
  // The RGB LED of the Nano 33 BLE is on when its pin is LOW.
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  SetLamp(LampState::kOff);
}

void SetLamp(LampState state) {
  switch (state) {
    case LampState::kOff:
      analogWrite(kLampPin, kDutyOff);
      break;
    case LampState::kLow:
      analogWrite(kLampPin, kDutyLow);
      break;
    case LampState::kOn:
      analogWrite(kLampPin, kDutyOn);
      break;
    case LampState::kBright:
      analogWrite(kLampPin, kDutyBright);
      break;
  }
}

void ShowListening(bool listening) {
  digitalWrite(LEDB, listening ? LOW : HIGH);
}

void ShowFailure(bool on) { digitalWrite(LEDR, on ? LOW : HIGH); }
