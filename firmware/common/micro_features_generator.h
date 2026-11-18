/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_

// From the micro_speech example of tinyMLx/arduino-library. Changed for this
// project: the functions return true or false instead of a TensorFlow Lite
// status and take no error reporter, so the same file builds with the
// TensorFlow Lite Micro versions of both boards and on the computer.

#include <stddef.h>
#include <stdint.h>

// Sets up any resources needed for the feature generation pipeline.
bool InitializeMicroFeatures();

// Converts audio sample data into a more compact form that's appropriate for
// feeding into a neural network.
bool GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read);

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_FEATURES_GENERATOR_H_
