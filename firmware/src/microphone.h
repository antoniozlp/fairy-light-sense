// Microphone gain and a level meter.
//
// The gain is the setting of the nRF52840's PDM input: 0 to 80 in steps of
// half a decibel, where 40 is neutral (0 dB). The micro_speech audio code
// starts the microphone at 20 (-10 dB); StartMicrophone() replaces that.

#ifndef FAIRY_LIGHT_SENSE_MICROPHONE_H_
#define FAIRY_LIGHT_SENSE_MICROPHONE_H_

#include <stdint.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"

// Gain at start-up. Override per build with -D MIC_GAIN=<0..80>, or change it
// while the board runs by sending "+" or "-" over the serial port.
#ifndef MIC_GAIN
#define MIC_GAIN 20
#endif

constexpr int kMicGainMin = 0;
constexpr int kMicGainMax = 80;
// One key press changes the gain by this much: 6 steps are 3 dB.
constexpr int kMicGainStep = 6;

// Starts audio capture and sets the gain.
void StartMicrophone(tflite::ErrorReporter* error_reporter, int gain);

// Sets the gain, limited to kMicGainMin..kMicGainMax, and returns what was set.
int SetMicGain(int gain);
int MicGain();
// The gain in decibels relative to the neutral setting.
float MicGainDb();

// Loudest sample between the two audio timestamps (at most the last 0.4 s),
// in decibels relative to full scale: 0 dB is clipping, speech in the training
// clips peaks around -10 dB.
float MicPeakDb(tflite::ErrorReporter* error_reporter, int32_t from_ms,
                int32_t to_ms);

#endif  // FAIRY_LIGHT_SENSE_MICROPHONE_H_
