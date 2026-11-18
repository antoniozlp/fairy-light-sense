// ESP32-S3-DevKitC-1: ESP-IDF, TensorFlow Lite Micro from Espressif's
// esp-tflite-micro component, an INMP441 microphone on I2S.

#ifndef FAIRY_LIGHT_SENSE_BOARD_H_
#define FAIRY_LIGHT_SENSE_BOARD_H_

#include <stdint.h>

constexpr char kBoardName[] = "esp32s3";

// This TensorFlow Lite Micro has no error reporter.
#define FAIRY_TFLM_ERROR_REPORTER 0

// Working memory of the interpreter. Generous, because this TensorFlow Lite
// Micro and its ESP-NN kernels may need more than the Nano's 30 kB for ds_cnn,
// and the ESP32-S3 has the RAM. The amount used is printed at start-up.
constexpr int kTensorArenaSize = 96 * 1024;

// --- Wiring of the INMP441 (see firmware/esp32s3/README.md) ---
// Change a pin with a build flag, for example -D MIC_PIN_SD=7.
#ifndef MIC_PIN_SCK
#define MIC_PIN_SCK 4  // bit clock, to SCK
#endif
#ifndef MIC_PIN_WS
#define MIC_PIN_WS 5  // word select, to WS
#endif
#ifndef MIC_PIN_SD
#define MIC_PIN_SD 6  // data, from SD
#endif
// The INMP441 sends on the left channel when its L/R pin is at GND, on the
// right channel when it is at 3V3.
#ifndef MIC_RIGHT_CHANNEL
#define MIC_RIGHT_CHANNEL 0
#endif

// The INMP441 has no gain setting; its samples are scaled instead. 40 keeps
// the upper 16 of its 24 bits. 64 is 12 dB more, the level Espressif's
// micro_speech example uses for this kind of microphone.
constexpr int kDefaultMicGain = 64;

// --- RGB LED ---
// GPIO38 on board revision v1.1, GPIO48 on v1.0. Both are driven, so either
// board lights up. Set one to -1 to use that pin for something else.
#ifndef RGB_LED_PIN
#define RGB_LED_PIN 38
#endif
#ifndef RGB_LED_PIN_V1_0
#define RGB_LED_PIN_V1_0 48
#endif

// --- The few Arduino calls main.cpp uses ---
// This build has no Arduino core. board.cpp provides them on top of ESP-IDF
// and calls setup() and loop().
class SerialPort {
 public:
  void begin(unsigned long) {}
  explicit operator bool() const { return true; }

  void print(const char* text);
  void print(int value);
  void print(unsigned int value);
  void print(long value);
  void print(unsigned long value);
  void print(double value, int digits = 2);
  void println(const char* text);
  void println(int value);

  // Characters typed in the serial monitor.
  int available();
  int read();

 private:
  int pending_ = -1;
};
extern SerialPort Serial;

uint32_t millis();
uint32_t micros();
void delay(uint32_t ms);

void setup();
void loop();

#endif  // FAIRY_LIGHT_SENSE_BOARD_H_
