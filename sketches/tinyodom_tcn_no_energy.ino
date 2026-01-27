// TinyODOM Arduino sketch variant without INA228 energy telemetry.
// Keeps the latency harness identical while skipping I2C sensor setup so
// larger tensor arenas/models fit on constrained boards.
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <cstdlib>
#include <cmath>

#include "model.h"

// TensorFlow Lite Micro headers required to mirror the desktop deployment.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Allow the optimizer or automation scripts to override deployment constants.
#ifndef TINYODOM_WINDOW_SIZE
#define TINYODOM_WINDOW_SIZE 200
#endif

#ifndef TINYODOM_NUM_CHANNELS
#define TINYODOM_NUM_CHANNELS 10
#endif

#ifndef TINYODOM_TENSOR_ARENA_BYTES
#define TINYODOM_TENSOR_ARENA_BYTES (25 * 1024)
#endif

namespace {

// Mirror TinyODOM deployment parameters (can be patched by hardware_utils).
constexpr int kWindowSize   = TINYODOM_WINDOW_SIZE;
constexpr int kNumChannels  = TINYODOM_NUM_CHANNELS;
constexpr size_t kTensorArenaSize = TINYODOM_TENSOR_ARENA_BYTES;
constexpr float kInvalidTelemetryValue = -1.0f;

// CMSIS-NN requires 16-byte alignment.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Globals reproduced from hello_world-style sketches.
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
uint32_t inference_seq = 0;

void EmitPowerTelemetryStub(uint32_t sequence_id) {
  Serial.print("inference seq: ");
  Serial.println(sequence_id);
  Serial.print("energy output (mJ): ");
  Serial.println(kInvalidTelemetryValue, 6);
  Serial.print("avg power output (mW): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("avg current output (mA): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("bus voltage output (V): ");
  Serial.println(kInvalidTelemetryValue, 3);
  Serial.print("idle power baseline (mW): ");
  Serial.println(kInvalidTelemetryValue, 3);
}

}  // namespace

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
  while (!Serial && millis() < 2000) {
    delay(10);
  }

  delay(1000);  // Wait for serial to settle.

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

  // Run several inferences back-to-back and average their latency (single timer span).
  const int kRuns = 10;
  int runs_completed = 0;
  delay(100);  // Ensure any prior serial output is complete.
  const uint32_t start_us = micros();
  for (int i = 0; i < kRuns; ++i) {
    const TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke() failed.");
      break;
    }
    runs_completed++;
  }
  const uint32_t total_latency_us = micros() - start_us;

  if (runs_completed > 0) {
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    ++inference_seq;
    // Emit the latency line expected by hardware_utils.py as all -1s
    EmitPowerTelemetryStub(inference_seq);
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
  }
}

void loop() {
  // Nothing else to do; keep the MCU alive for serial viewing.
  delay(1000);
}
