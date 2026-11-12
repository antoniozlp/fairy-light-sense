// What the board shows. For now the lamp is the orange built-in LED and the
// blue LED shows that a command is awaited. The LED string comes later, on
// kLampPin through a MOSFET.

#ifndef FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_
#define FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_

enum class LampState { kOff, kLow, kOn, kBright };

void SetupLampOutput();
void SetLamp(LampState state);
void ShowListening(bool listening);
const char* LampStateName(LampState state);

#endif  // FAIRY_LIGHT_SENSE_LAMP_OUTPUT_H_
