// ESP-IDF side of board.h: the serial port, the clock, and the main task that
// runs setup() and loop() of main.cpp.

#include "board.h"

#include <stdio.h>
#include <unistd.h>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

SerialPort Serial;

// The console of ESP-IDF is UART0, the USB port labelled "UART" on the board.
// Text is sent when a line is complete.
void SerialPort::print(const char* text) { fputs(text, stdout); }
void SerialPort::print(int value) { printf("%d", value); }
void SerialPort::print(unsigned int value) { printf("%u", value); }
void SerialPort::print(long value) { printf("%ld", value); }
void SerialPort::print(unsigned long value) { printf("%lu", value); }
void SerialPort::print(double value, int digits) {
  printf("%.*f", digits, value);
}
void SerialPort::println(const char* text) { puts(text); }
void SerialPort::println(int value) { printf("%d\n", value); }

// Reading the console does not wait: without a character it returns at once.
int SerialPort::available() {
  if (pending_ < 0) {
    unsigned char c = 0;
    if (::read(STDIN_FILENO, &c, 1) == 1) {
      pending_ = c;
    }
  }
  return pending_ >= 0 ? 1 : 0;
}

int SerialPort::read() {
  if (available() == 0) {
    return -1;
  }
  const int c = pending_;
  pending_ = -1;
  return c;
}

uint32_t millis() { return static_cast<uint32_t>(esp_timer_get_time() / 1000); }
uint32_t micros() { return static_cast<uint32_t>(esp_timer_get_time()); }
void delay(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }

extern "C" void app_main() {
  setup();
  while (true) {
    loop();
    // One tick (1 ms) for the other tasks. Without it the idle task never
    // runs and the task watchdog complains.
    vTaskDelay(1);
  }
}
