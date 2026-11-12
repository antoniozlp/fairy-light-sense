// Voice-controlled fairy light lamp for the Arduino Nano 33 BLE Sense.
//
// The loop follows the micro_speech example of TensorFlow Lite for
// Microcontrollers: take the newest audio, update the 49 x 40 feature map of
// the last second, run the model, and hand the scores to the recogniser. Say
// the wake word, then a command.

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <string.h>

#include "audio_provider.h"
#include "feature_provider.h"
#include "lamp_output.h"
#include "lamp_recognizer.h"
#include "micro_features_micro_model_settings.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {

// Print a line with timing and scores every few seconds. Useful while bringing
// the board up; set to 0 to print detections only.
constexpr int32_t kStatusEveryMs = 2000;

// Working memory of the interpreter. ds_cnn needs the most; the amount actually
// used is printed at start-up.
constexpr int kTensorArenaSize = 48 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
FeatureProvider* feature_provider = nullptr;
LampRecognizer recognizer;
LampState lamp = LampState::kOff;

int8_t feature_buffer[kFeatureElementCount];
int32_t previous_time = 0;
// Why set-up stopped; nullptr once everything is ready.
const char* failure = "set-up did not finish";

// Timing, for the status line.
int32_t last_status_time = 0;
uint32_t inference_micros = 0;
uint32_t loop_count = 0;

void PrintDetection(const Detection& detection, int32_t now_ms) {
  Serial.print(now_ms / 1000.0f, 1);
  Serial.print(" s  ");
  Serial.print(kCategoryLabels[detection.label]);
  Serial.print(" ");
  Serial.print(detection.score, 2);
  Serial.print("  ->  ");
  if (detection.label == kWakeWordIndex) {
    Serial.println("listening for a command");
  } else if (detection.accepted) {
    Serial.print("lamp ");
    Serial.println(LampStateName(lamp));
  } else {
    Serial.print("ignored, say \"");
    Serial.print(kCategoryLabels[kWakeWordIndex]);
    Serial.println("\" first");
  }
}

void PrintStatus(int32_t now_ms) {
  const int32_t elapsed = now_ms - last_status_time;
  // The model name leads every status line, so a log shows which model was running.
  Serial.print("[");
  Serial.print(kModelName);
  Serial.print("] ");
  Serial.print(loop_count * 1000.0f / elapsed, 1);
  Serial.print(" classifications/s, inference ");
  Serial.print(inference_micros / 1000.0f / (loop_count ? loop_count : 1), 1);
  Serial.print(" ms |");
  for (int i = kWakeWordIndex; i < kCategoryCount; ++i) {
    Serial.print(" ");
    Serial.print(kCategoryLabels[i]);
    Serial.print(" ");
    Serial.print(recognizer.average()[i], 2);
  }
  Serial.print(" | lamp ");
  Serial.println(LampStateName(lamp));
  last_status_time = now_ms;
  inference_micros = 0;
  loop_count = 0;
}

}  // namespace

void setup() {
  Serial.begin(9600);
  SetupLampOutput();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    failure = "the model has an unsupported schema version";
    return;
  }

  // The operations of both our architectures (tiny_conv and ds_cnn), so either
  // model runs without touching this list. The training notebook prints the
  // operations of each model.
  static tflite::MicroMutableOpResolver<6> op_resolver(error_reporter);
  if (op_resolver.AddConv2D() != kTfLiteOk ||
      op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      op_resolver.AddAveragePool2D() != kTfLiteOk ||
      op_resolver.AddFullyConnected() != kTfLiteOk ||
      op_resolver.AddReshape() != kTfLiteOk ||
      op_resolver.AddSoftmax() != kTfLiteOk) {
    failure = "could not register the model's operations";
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    failure = "AllocateTensors() failed: unsupported operation or arena too small";
    return;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  if (model_input->type != kTfLiteInt8 ||
      model_input->bytes != static_cast<size_t>(kFeatureElementCount) ||
      model_output->type != kTfLiteInt8 ||
      model_output->bytes != static_cast<size_t>(kCategoryCount)) {
    failure = "the model's input or output does not match the settings";
    return;
  }

  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  // Give a serial monitor up to four seconds to connect, so it sees the banner.
  while (!Serial && millis() < 4000) {
  }
  delay(500);  // text sent in the instant of connecting gets lost
  Serial.print("fairy-light-sense, model ");
  Serial.print(kModelName);
  Serial.print(", ");
  Serial.print(g_model_len);
  Serial.print(" bytes, tensor arena ");
  Serial.print(static_cast<int>(interpreter->arena_used_bytes()));
  Serial.print(" of ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes used");
  Serial.print("Say \"");
  Serial.print(kCategoryLabels[kWakeWordIndex]);
  Serial.println("\", then a command.");
  failure = nullptr;
}

void loop() {
  if (failure != nullptr) {
    // Set-up failed: blink red and keep saying why.
    digitalWrite(LEDR, LOW);
    delay(300);
    digitalWrite(LEDR, HIGH);
    delay(1700);
    Serial.print("Set-up failed: ");
    Serial.println(failure);
    return;
  }

  // Update the feature map with the audio that arrived since the last loop.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  if (feature_provider->PopulateFeatureData(error_reporter, previous_time,
                                            current_time,
                                            &how_many_new_slices) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  if (how_many_new_slices == 0) {
    return;
  }

  memcpy(model_input->data.int8, feature_buffer, kFeatureElementCount);
  const uint32_t start = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  inference_micros += micros() - start;
  ++loop_count;

  // int8 scores back to probabilities, with the scale stored in the model.
  float scores[kCategoryCount];
  for (int i = 0; i < kCategoryCount; ++i) {
    scores[i] = (model_output->data.int8[i] - model_output->params.zero_point) *
                model_output->params.scale;
  }

  Detection detection;
  if (recognizer.Update(scores, current_time, &detection)) {
    if (detection.label != kWakeWordIndex && detection.accepted) {
      // Categories after the wake word are the commands, in label order.
      const char* word = kCategoryLabels[detection.label];
      if (strcmp(word, "on") == 0) {
        lamp = LampState::kOn;
      } else if (strcmp(word, "off") == 0) {
        lamp = LampState::kOff;
      } else if (strcmp(word, "bright") == 0) {
        lamp = LampState::kBright;
      } else if (strcmp(word, "low") == 0) {
        lamp = LampState::kLow;
      }
      SetLamp(lamp);
    }
    PrintDetection(detection, current_time);
  }
  ShowListening(recognizer.Listening(current_time));

  if (kStatusEveryMs > 0 && current_time - last_status_time >= kStatusEveryMs) {
    PrintStatus(current_time);
  }
}
