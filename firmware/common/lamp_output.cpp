// The part of lamp_output.h that is the same on every board.

#include "lamp_output.h"

const char* LampStateName(LampState state) {
  switch (state) {
    case LampState::kOff:
      return "OFF";
    case LampState::kLow:
      return "LOW";
    case LampState::kOn:
      return "ON";
    case LampState::kBright:
      return "BRIGHT";
  }
  return "?";
}
