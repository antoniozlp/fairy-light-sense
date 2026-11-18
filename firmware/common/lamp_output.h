// What the board shows: the lamp state, whether a command is awaited, and a
// failed set-up. Each board has its own lamp_output.cpp for its LEDs. The LED
// string comes later, through a MOSFET.

#ifndef FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_
#define FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_

enum class LampState { kOff, kLow, kOn, kBright };

void SetupLampOutput();
void SetLamp(LampState state);
void ShowListening(bool listening);
// Red light for a failed set-up; the caller blinks it.
void ShowFailure(bool on);
const char* LampStateName(LampState state);

#endif  // FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_
