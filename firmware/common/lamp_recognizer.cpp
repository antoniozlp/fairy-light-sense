#include "lamp_recognizer.h"

bool LampRecognizer::Update(const float* scores, int32_t now_ms,
                            Detection* detection) {
  Entry& entry = history_[history_next_];
  entry.time_ms = now_ms;
  for (int i = 0; i < kCategoryCount; ++i) {
    entry.scores[i] = scores[i];
  }
  history_next_ = (history_next_ + 1) % kHistorySize;
  if (history_count_ < kHistorySize) {
    ++history_count_;
  }

  // Average the classifications of the last kAverageMs. When the model is slow
  // only one may fall inside that time; then that one decides.
  int used = 0;
  for (int i = 0; i < kCategoryCount; ++i) {
    average_[i] = 0.0f;
  }
  for (int n = 0; n < history_count_; ++n) {
    const Entry& past = history_[n];
    if (past.time_ms > now_ms - kAverageMs) {
      for (int i = 0; i < kCategoryCount; ++i) {
        average_[i] += past.scores[i];
      }
      ++used;
    }
  }
  int top = 0;
  for (int i = 0; i < kCategoryCount; ++i) {
    average_[i] /= used;
    if (average_[i] > average_[top]) {
      top = i;
    }
  }

  if (top == kSilenceIndex || top == kUnknownIndex ||
      average_[top] < kThreshold) {
    return false;
  }
  if (heard_before_[top] && now_ms - last_heard_ms_[top] < kSuppressionMs) {
    return false;
  }
  heard_before_[top] = true;
  last_heard_ms_[top] = now_ms;

  detection->label = top;
  detection->score = average_[top];
  if (top == kWakeWordIndex) {
    listening_until_ms_ = now_ms + kCommandWindowMs;
    detection->accepted = true;
  } else {
    detection->accepted = Listening(now_ms);
    listening_until_ms_ = kNever;
  }
  return true;
}
