// Microphone gain and a level meter, the same on every board.
//
// The gain goes from 0 to 80 in steps of half a decibel; 40 is neutral (0 dB).
// On the Nano 33 BLE Sense it is the setting of the nRF52840's PDM input. On
// the ESP32-S3 the samples of the I2S microphone are scaled by that amount.

#ifndef FAIRY_LIGHT_SENSE_MICROPHONE_H_
#define FAIRY_LIGHT_SENSE_MICROPHONE_H_

#include <stdint.h>

#include "board.h"

// Gain at start-up: the board's default (kDefaultMicGain in its board.h),
// unless the build sets -D MIC_GAIN=<0..80>. While the board runs, "+" and "-"
// over the serial port change it.
#ifndef MIC_GAIN
#define MIC_GAIN kDefaultMicGain
#endif

constexpr int kMicGainMin = 0;
constexpr int kMicGainMax = 80;
// One key press changes the gain by this much: 6 steps are 3 dB.
constexpr int kMicGainStep = 6;

// Starts audio capture and sets the gain. False if the microphone did not start.
bool StartMicrophone(int gain);

// Sets the gain, limited to kMicGainMin..kMicGainMax, and returns what was set.
int SetMicGain(int gain);
int MicGain();
// The gain in decibels relative to the neutral setting.
float MicGainDb();

// Loudest sample between the two audio timestamps (at most the last 0.4 s),
// in decibels relative to full scale: 0 dB is clipping, speech in the training
// clips peaks around -10 dB.
float MicPeakDb(int32_t from_ms, int32_t to_ms);

#endif  // FAIRY_LIGHT_SENSE_MICROPHONE_H_
