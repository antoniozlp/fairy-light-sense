// Arduino Nano 33 BLE Sense: Arduino core, TensorFlow Lite Micro 2.4 from the
// Harvard_TinyMLx library, on-board PDM microphone.

#ifndef FAIRY_LIGHT_SENSE_BOARD_H_
#define FAIRY_LIGHT_SENSE_BOARD_H_

#include <Arduino.h>
#include <TensorFlowLite.h>

constexpr char kBoardName[] = "nano33ble";

// This TensorFlow Lite Micro still passes an error reporter to the interpreter.
#define FAIRY_TFLM_ERROR_REPORTER 1

// Working memory of the interpreter. ds_cnn needs the most: 30 kB.
constexpr int kTensorArenaSize = 48 * 1024;

// PDM gain of the micro_speech example: -10 dB.
constexpr int kDefaultMicGain = 20;

#endif  // FAIRY_LIGHT_SENSE_BOARD_H_
