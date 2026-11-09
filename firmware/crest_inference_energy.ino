// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// CREST Arduino sketch that mirrors the original Mbed latency harness.
// It seeds the TensorFlow Lite Micro interpreter, populates the windowed
// sensor buffer with synthetic data, runs one inference, and prints the
// measured latency in the format expected by the HIL tooling.
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <cstdlib>
#include <cmath>

#include "model.h"
#include "common/crest_hil_config.h"
#include "common/crest_power.h"
#include "common/crest_clock_telemetry.h"

// TensorFlow Lite Micro headers required to mirror the desktop deployment.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Allow the optimizer or automation scripts to override deployment constants.
#ifndef CREST_WINDOW_SIZE
#define CREST_WINDOW_SIZE 200
#endif

#ifndef CREST_NUM_CHANNELS
#define CREST_NUM_CHANNELS 10
#endif

#ifndef CREST_TENSOR_ARENA_BYTES
#define CREST_TENSOR_ARENA_BYTES (25 * 1024)
#endif

// Optional runtime controls for targets that cannot rely on host serial
// handshakes (for example Portenta CM4 harness-only runs).
#ifndef CREST_AUTOSTART
#define CREST_AUTOSTART 0
#endif

// Delay before inference when autostart is enabled. This gives upload/reset
// and harness setup enough time to settle before the measurement pulse.
#ifndef CREST_AUTOSTART_DELAY_MS
#define CREST_AUTOSTART_DELAY_MS 0
#endif

// Skip waiting on `while (!Serial)` for targets where host USB CDC may be
// unavailable/restricted at runtime, avoiding setup-time stalls.
#ifndef CREST_SKIP_SERIAL_WAIT
#define CREST_SKIP_SERIAL_WAIT 0
#endif

namespace {

// Mirror CREST deployment parameters (can be patched by hardware_utils).
constexpr int kWindowSize   = CREST_WINDOW_SIZE;
constexpr int kNumChannels  = CREST_NUM_CHANNELS;
constexpr size_t kTensorArenaSize = CREST_TENSOR_ARENA_BYTES;

// CMSIS-NN requires 16-byte alignment.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Globals reproduced from hello_world-style sketches.
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

uint32_t inference_seq = 0;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;

}  // namespace

bool WaitForStartCommand(uint32_t timeout_ms) {
  // Poll for a host START command without blocking indefinitely.
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

// Populate the model input once with random data (0..5) like the legacy Mbed app.
// This keeps the Arduino sketch compatible with the hardware_utils HIL pipeline.
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
  // Set up serial so the latency prints can be collected by the host.
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !CREST_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif

  delay(1000);  // Wait for serial to settle.

  // Initialize the trigger pin used by the harness.
  pinMode(CREST_HARNESS_TRIGGER_PIN, OUTPUT);
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, LOW);
  // Keep harness disarmed until this DUT is ready to run inference.
  pinMode(CREST_HARNESS_ARM_PIN, OUTPUT);
  digitalWrite(CREST_HARNESS_ARM_PIN, HIGH);

  // Serial.print("kTensorArenaSize: ");
  // Serial.println(kTensorArenaSize);

  // Initialize INA228 only when enabled (no-op on DUT builds).
  crest_power::InitializePowerMonitor();

  // Initialize the MCU-specific hooks required by TFLM.
  tflite::InitializeTarget();
  error_reporter = tflite::GetMicroErrorReporter();

  // Map the compiled model.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d != supported %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    while (true) {
      delay(100);
    }
  }

  // Removed MicroProfiler for Arduino sketch to reduce complexity.
  // Pull in all ops so the optimizer can select freely.
  static tflite::AllOpsResolver resolver;

  // Build the interpreter with the same signature used in hello_world.
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

  // If autostart is enabled, run without START handshake (used by harness-only
  // runtime flows). Otherwise use the standard host READY/START protocol.
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

  // Run several inferences back-to-back and average their latency (single timer span).
  const int kRuns = CREST_INFERENCE_RUNS;
  int runs_completed = 0;
  // Arm the harness first, then hold low long enough for stable-low detection.
  digitalWrite(CREST_HARNESS_ARM_PIN, LOW);
  delay(CREST_DUT_ARM_HOLD_MS);
  // Start the measurement window and signal the harness.
  crest_power::ResetAccumulators();
  crest_clock::ConfigureCycleCounter();
  delay(100);  // Ensure any prior serial output is complete.
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, HIGH);
  const uint32_t start_us = micros();
  const uint32_t start_cycles = crest_clock::ReadCycleCounterRaw();
  for (int i = 0; i < kRuns; ++i) {
    const TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke() failed.");
      break;
    }
    runs_completed++;
  }
  const uint32_t end_cycles = crest_clock::ReadCycleCounterRaw();
  const uint32_t total_latency_us = micros() - start_us;
  // Signal the harness that inference is complete.
  digitalWrite(CREST_HARNESS_TRIGGER_PIN, LOW);
  // Disarm harness after pulse completion.
  digitalWrite(CREST_HARNESS_ARM_PIN, HIGH);

  if (runs_completed > 0) {
    const float energy_total_j = crest_power::FlushEnergyWindow();
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    const float clock_hz = crest_clock::ReadClockHz();
    const float dwt_cycles_per_inference = crest_clock::ComputeCyclesPerInference(
        start_cycles, end_cycles, runs_completed, total_latency_us, clock_hz);
    ++inference_seq;
    // Emit run count + telemetry for host parsing.
    Serial.print("runs: ");
    Serial.println(runs_completed);
    crest_power::EmitPowerTelemetry(
        inference_seq, latency_s, runs_completed,
        crest_power::GetIdleBaselinePower(), energy_total_j);
    crest_clock::EmitClockTelemetry(clock_hz, dwt_cycles_per_inference);
    // Emit the latency line expected by hardware_utils.py.
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
  }
}

void loop() {
  // Nothing else to do; keep the MCU alive for serial viewing.
  delay(1000);
}
