// Cadenced phase sketch for cadenced Portenta analysis.
// This runs one invoke per latency budget slot and idles with light sleep
// while waiting for the next slot.
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <cstdlib>
#include <cmath>

#include "model.h"
#include "common/tinyodom_hil_config.h"
#include "common/tinyodom_power.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TINYODOM_WINDOW_SIZE
#define TINYODOM_WINDOW_SIZE 200
#endif

#ifndef TINYODOM_NUM_CHANNELS
#define TINYODOM_NUM_CHANNELS 10
#endif

#ifndef TINYODOM_TENSOR_ARENA_BYTES
#define TINYODOM_TENSOR_ARENA_BYTES (25 * 1024)
#endif

#ifndef TINYODOM_LATENCY_BUDGET_MS
#define TINYODOM_LATENCY_BUDGET_MS 200
#endif

#ifndef TINYODOM_AUTOSTART
#define TINYODOM_AUTOSTART 0
#endif

#ifndef TINYODOM_AUTOSTART_DELAY_MS
#define TINYODOM_AUTOSTART_DELAY_MS 0
#endif

#ifndef TINYODOM_SKIP_SERIAL_WAIT
#define TINYODOM_SKIP_SERIAL_WAIT 0
#endif

namespace {

constexpr int kWindowSize = TINYODOM_WINDOW_SIZE;
constexpr int kNumChannels = TINYODOM_NUM_CHANNELS;
constexpr size_t kTensorArenaSize = TINYODOM_TENSOR_ARENA_BYTES;
constexpr uint32_t kCadenceUs =
    static_cast<uint32_t>(TINYODOM_LATENCY_BUDGET_MS) * 1000UL;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

uint32_t inference_seq = 0;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;

inline bool TimeReached(uint32_t now_us, uint32_t target_us) {
  return static_cast<int32_t>(now_us - target_us) >= 0;
}

inline void LightSleepUntil(uint32_t target_us) {
  while (!TimeReached(micros(), target_us)) {
    __WFI();
  }
}

}  // namespace

bool WaitForStartCommand(uint32_t timeout_ms) {
  const uint32_t start_ms = millis();
  while (millis() - start_ms < timeout_ms) {
    if (!Serial.available()) {
      delay(5);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.equalsIgnoreCase(TINYODOM_CMD_START)) {
      return true;
    }
  }
  return false;
}

void FillInputTensor() {
  const float scale = input->params.scale;
  const int32_t zero_point = input->params.zero_point;

  for (int sample = 0; sample < kWindowSize; ++sample) {
    for (int channel = 0; channel < kNumChannels; ++channel) {
      const int offset = sample * kNumChannels + channel;
      const float value =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0f;

      if (input->type == kTfLiteFloat32) {
        input->data.f[offset] = value;
      } else if (input->type == kTfLiteInt8) {
        int32_t quantized = static_cast<int32_t>(roundf(value / scale)) + zero_point;
        quantized = min(127, max(-128, quantized));
        input->data.int8[offset] = static_cast<int8_t>(quantized);
      } else {
        error_reporter->Report("Unsupported input tensor type (%d).", input->type);
        return;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !TINYODOM_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif

  delay(1000);

  pinMode(TINYODOM_HARNESS_TRIGGER_PIN, OUTPUT);
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, LOW);
  pinMode(TINYODOM_HARNESS_ARM_PIN, OUTPUT);
  digitalWrite(TINYODOM_HARNESS_ARM_PIN, HIGH);

  tinyodom_power::InitializePowerMonitor();

  tflite::InitializeTarget();
  error_reporter = tflite::GetMicroErrorReporter();

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d != supported %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    while (true) {
      delay(100);
    }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, /*resource_variables=*/nullptr,
      /*profiler=*/nullptr);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed. Tensor arena bytes: %u",
                           static_cast<unsigned>(kTensorArenaSize));
    while (true) {
      delay(100);
    }
  }

  input = interpreter->input(0);
  FillInputTensor();

#if TINYODOM_AUTOSTART
  if (TINYODOM_AUTOSTART_DELAY_MS > 0) {
    delay(TINYODOM_AUTOSTART_DELAY_MS);
  }
#else
  Serial.println(TINYODOM_DUT_READY);
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println(TINYODOM_DUT_READY);
  }
#endif

  const int kRuns = TINYODOM_INFERENCE_RUNS;
  int runs_completed = 0;
  int deadline_miss_count = 0;

  digitalWrite(TINYODOM_HARNESS_ARM_PIN, LOW);
  delay(TINYODOM_DUT_ARM_HOLD_MS);
  tinyodom_power::ResetAccumulators();
  delay(100);
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, HIGH);

  const uint32_t phase_start_us = micros();
  uint32_t next_release_us = phase_start_us;

  for (int i = 0; i < kRuns; ++i) {
    LightSleepUntil(next_release_us);
    const uint32_t invoke_start_us = micros();
    const TfLiteStatus invoke_status = interpreter->Invoke();
    const uint32_t invoke_end_us = micros();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke() failed.");
      break;
    }
    runs_completed++;
    const uint32_t next_deadline_us = next_release_us + kCadenceUs;
    if (TimeReached(invoke_end_us, next_deadline_us)) {
      deadline_miss_count++;
    }
    next_release_us = next_deadline_us;
  }

  // Include the full final cadence slot in the measurement window so
  // total phase latency represents kRuns * latency_budget when deadlines
  // are met. If we're already behind schedule, this returns immediately.
  if (runs_completed == kRuns) {
    LightSleepUntil(next_release_us);
  }

  const uint32_t total_latency_us = micros() - phase_start_us;
  digitalWrite(TINYODOM_HARNESS_TRIGGER_PIN, LOW);
  digitalWrite(TINYODOM_HARNESS_ARM_PIN, HIGH);

  if (runs_completed > 0) {
    const float energy_total_j = tinyodom_power::FlushEnergyWindow();
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    ++inference_seq;
    Serial.print("runs: ");
    Serial.println(runs_completed);
    tinyodom_power::EmitPowerTelemetry(
        inference_seq, latency_s, runs_completed,
        tinyodom_power::GetIdleBaselinePower(), energy_total_j);
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
    Serial.print("cadence deadline misses: ");
    Serial.println(deadline_miss_count);
    Serial.print("cadence budget (ms): ");
    Serial.println(TINYODOM_LATENCY_BUDGET_MS);
  }
}

void loop() {
  delay(1000);
}
