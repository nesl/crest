// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// CREST Arduino cadenced runtime sketch.
// This variant keeps INA228 energy telemetry enabled and spreads invokes across
// a fixed cadence so the reported window includes both inference work and the
// scheduled gaps between releases.
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <cstdlib>
#include <cmath>

#include "model.h"
#include "common/crest_hil_config.h"
#include "common/crest_power.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Allow deployment scripts to override the model dimensions without editing the
// sketch source directly.
#ifndef CREST_WINDOW_SIZE
#define CREST_WINDOW_SIZE 200
#endif

#ifndef CREST_NUM_CHANNELS
#define CREST_NUM_CHANNELS 10
#endif

#ifndef CREST_TENSOR_ARENA_BYTES
#define CREST_TENSOR_ARENA_BYTES (25 * 1024)
#endif

// Cadenced mode schedules one invoke per latency-budget slot.
#ifndef CREST_LATENCY_BUDGET_MS
#define CREST_LATENCY_BUDGET_MS 200
#endif

// Match the no-energy sketch's host handshake while still allowing harness-only
// autostart flows on targets with limited USB CDC support.
#ifndef CREST_AUTOSTART
#define CREST_AUTOSTART 0
#endif

#ifndef CREST_AUTOSTART_DELAY_MS
#define CREST_AUTOSTART_DELAY_MS 0
#endif

#ifndef CREST_SKIP_SERIAL_WAIT
#define CREST_SKIP_SERIAL_WAIT 0
#endif

namespace {

// Mirror the deployment constants patched in by the hardware pipeline.
constexpr int kWindowSize = CREST_WINDOW_SIZE;
constexpr int kNumChannels = CREST_NUM_CHANNELS;
constexpr size_t kTensorArenaSize = CREST_TENSOR_ARENA_BYTES;
// Convert the requested cadence to microseconds once so the release loop can
// compare against `micros()` directly.
constexpr uint32_t kCadenceUs =
    static_cast<uint32_t>(CREST_LATENCY_BUDGET_MS) * 1000UL;
// CMSIS-NN kernels expect 16-byte alignment.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Globals mirror the TensorFlow Lite Micro hello_world setup so the deployment
// path stays close to the standard embedded examples.
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

uint32_t inference_seq = 0;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;

// Treat unsigned wraparound correctly when comparing two `micros()` samples.
inline bool TimeReached(uint32_t now_us, uint32_t target_us) {
  return static_cast<int32_t>(now_us - target_us) >= 0;
}

// Sleep lightly until the next cadence release so idle time is still measured
// inside the overall slot budget.
inline void LightSleepUntil(uint32_t target_us) {
  while (!TimeReached(micros(), target_us)) {
    __WFI();
  }
}

}  // namespace

bool WaitForStartCommand(uint32_t timeout_ms) {
  // Poll for a host START command without blocking forever if the host misses
  // the first READY line after reset.
  const uint32_t start_ms = millis();
  while (millis() - start_ms < timeout_ms) {
    if (!Serial.available()) {
      delay(5);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.equalsIgnoreCase(CREST_CMD_START)) {
      return true;
    }
  }
  return false;
}

// Populate the input tensor with the same synthetic range used by the baseline
// sketch so hardware comparisons stay aligned across variants.
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
        // Reuse the model's quantization parameters so both float and int8
        // deployments receive comparable synthetic input values.
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
  // Bring up serial first so READY and telemetry prints have somewhere to go.
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !CREST_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif

  delay(1000);

  // Initialize the shared trigger/arm GPIOs used by the measurement harness.
  pinMode(CREST_HARNESS_TRIGGER_PIN, OUTPUT);
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, LOW);
  pinMode(CREST_HARNESS_ARM_PIN, OUTPUT);
  digitalWrite(CREST_HARNESS_ARM_PIN, HIGH);

  // Bring up the power monitor before inference so the idle baseline and
  // accumulator state start from a known point.
  crest_power::InitializePowerMonitor();

  // Initialize the MCU-specific TensorFlow Lite Micro support.
  tflite::InitializeTarget();
  error_reporter = tflite::GetMicroErrorReporter();

  // Map the compiled TFLite flatbuffer and verify schema compatibility.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d != supported %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    while (true) {
      delay(100);
    }
  }

  // Pull in all ops so the search pipeline can deploy arbitrary candidates
  // without maintaining a handwritten resolver list.
  static tflite::AllOpsResolver resolver;
  // Build the interpreter with the same basic structure used by upstream
  // hello_world examples.
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

  // Either wait for the host handshake or autostart directly for harness-only
  // flows that cannot rely on USB serial coordination.
#if CREST_AUTOSTART
  if (CREST_AUTOSTART_DELAY_MS > 0) {
    delay(CREST_AUTOSTART_DELAY_MS);
  }
#else
  Serial.println(CREST_DUT_READY);
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println(CREST_DUT_READY);
  }
#endif

  const int kRuns = CREST_INFERENCE_RUNS;
  int runs_completed = 0;

  // Arm the harness and clear any stale power accumulators immediately before
  // entering the measured cadence window.
  digitalWrite(CREST_HARNESS_ARM_PIN, LOW);
  delay(CREST_DUT_ARM_HOLD_MS);
  crest_power::ResetAccumulators();
  delay(100);
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, HIGH);

  // Measure the full scheduled window, not just active invoke time. That means
  // the timer output intentionally includes the light-sleep gaps between slots.
  const uint32_t phase_start_us = micros();
  uint32_t next_release_us = phase_start_us;

  for (int i = 0; i < kRuns; ++i) {
    // Wait until the next release point so each invoke starts on the requested
    // cadence boundary.
    LightSleepUntil(next_release_us);
    const TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke() failed.");
      break;
    }
    runs_completed++;
    next_release_us += kCadenceUs;
  }

  // If every scheduled invoke ran, extend the measurement window through the
  // trailing edge of the final slot as well.
  if (runs_completed == kRuns) {
    LightSleepUntil(next_release_us);
  }

  const uint32_t total_latency_us = micros() - phase_start_us;
  // Drop the trigger and disarm the harness once the cadenced window is over.
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, LOW);
  digitalWrite(CREST_HARNESS_ARM_PIN, HIGH);

  if (runs_completed > 0) {
    // Flush energy after the full cadence window so the per-slot totals include
    // both the inference work and the scheduled idle portion.
    const float energy_total_j = crest_power::FlushEnergyWindow();
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    ++inference_seq;
    Serial.print("runs: ");
    Serial.println(runs_completed);
    crest_power::EmitPowerTelemetry(
        inference_seq, latency_s, runs_completed,
        crest_power::GetIdleBaselinePower(), energy_total_j);
    // For this cadenced Arduino sketch, timer output is per-slot wall time and
    // intentionally includes the scheduled wait portion.
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
  }
}

void loop() {
  delay(1000);
}
