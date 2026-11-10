// Turns the stream of model scores into detections and applies the lamp's rule:
// the wake word first, then one command within a short time.
// Same logic and defaults as `Recognizer` in src/fairy_kws/listen.py.

#ifndef FAIRY_LIGHT_SENSE_LAMP_RECOGNIZER_H_
#define FAIRY_LIGHT_SENSE_LAMP_RECOGNIZER_H_

#include <stdint.h>

#include "model_data.h"

struct Detection {
  int label;      // index into kCategoryLabels
  float score;    // averaged score, 0..1
  bool accepted;  // false for a command heard without the wake word
};

class LampRecognizer {
 public:
  // Score a keyword needs to count.
  static constexpr float kThreshold = 0.7f;
  // Scores are averaged over this time.
  static constexpr int32_t kAverageMs = 300;
  // The same word is not reported again within this time: one spoken word stays
  // inside the one-second window for several classifications in a row.
  static constexpr int32_t kSuppressionMs = 1500;
  // A command is accepted for this long after the wake word.
  static constexpr int32_t kCommandWindowMs = 2000;

  // Adds the scores (one per category, 0..1) of a classification made at
  // `now_ms`. Returns true and fills `detection` when a keyword is detected.
  bool Update(const float* scores, int32_t now_ms, Detection* detection);

  // True while a command would be accepted.
  bool Listening(int32_t now_ms) const { return now_ms <= listening_until_ms_; }

  // Latest averaged scores, one per category.
  const float* average() const { return average_; }

 private:
  static constexpr int kHistorySize = 8;
  static constexpr int32_t kNever = -1000000;

  struct Entry {
    int32_t time_ms;
    float scores[kCategoryCount];
  };

  Entry history_[kHistorySize] = {};
  int history_count_ = 0;
  int history_next_ = 0;
  float average_[kCategoryCount] = {};
  int32_t last_heard_ms_[kCategoryCount] = {};
  bool heard_before_[kCategoryCount] = {};
  int32_t listening_until_ms_ = kNever;
};

#endif  // FAIRY_LIGHT_SENSE_LAMP_RECOGNIZER_H_
