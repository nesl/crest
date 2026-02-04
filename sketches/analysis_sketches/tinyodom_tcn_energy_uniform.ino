// TinyODOM Arduino sketch that mirrors the original Mbed latency harness.
// It seeds the TensorFlow Lite Micro interpreter, populates the windowed
// sensor buffer with synthetic data, runs one inference, and prints the
// measured latency in the format expected by the HIL tooling.
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <Wire.h>
#include <Adafruit_INA228.h>
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

// CMSIS-NN requires 16-byte alignment.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Globals reproduced from hello_world-style sketches.
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

Adafruit_INA228 power_monitor;
bool power_monitor_ready = false;
uint32_t inference_seq = 0;

// On-board Adafruit INA228 breakout precision shunt value (15 milliohms).
// Using the actual resistor improves calibration vs the previous placeholder (0.25Ω).
// Typical Nano 33 BLE Sense draw during inference <120mA, so we keep max_current at 0.30A
// to retain headroom while achieving a current LSB of ≈0.572 µA (0.30 / 2^19).
constexpr float kShuntResistanceOhms = 0.015f;
constexpr float kMaxExpectedCurrentAmps = 0.30f;
// Higher averaging for ~200ms per-inference windows: cleaner, lower-noise samples.
// Library 3.x only exposes 1/4/16/64/128... counts, so we use 64 (≈27.5 ms per frame).
constexpr INA228_ConversionTime kCurrentConversionTime = INA228_TIME_280_us;
constexpr INA228_ConversionTime kVoltageConversionTime = INA228_TIME_150_us;
// Averaging count for ADC measurements.
// Library 3.x exposes counts of 1/4/16/64/128/256/512/1024, so we pick 64 (~27 ms window).
constexpr INA228_AveragingCount kAveragingCount = INA228_COUNT_64;
constexpr uint16_t kTriggeredFlushDelayMs = 20;
constexpr float kInvalidTelemetryValue = -1.0f;
float idle_baseline_power_mw = kInvalidTelemetryValue;

#ifndef INA228_ADCRANGE_40_96MV
#define INA228_ADCRANGE_40_96MV 1
#endif

#ifdef INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP
constexpr auto kTriggerStopMode = INA228_MODE_TRIGGERED_SHUNT_BUS_TEMP;
#else
constexpr auto kTriggerStopMode =
    static_cast<decltype(INA228_MODE_CONTINUOUS)>(0x7);
#endif

// Sanitize float value for telemetry output.
float SanitizeFloat(float value) {
  if (isnan(value) || isinf(value)) {
    return kInvalidTelemetryValue;
  }
  return value;
}

bool InitializePowerMonitor() {
  if (power_monitor_ready) {
    return true;
  }
  if (!power_monitor.begin()) {
    Serial.println("energy error: init_failed");
    return false;
  }
  power_monitor.setShunt(kShuntResistanceOhms, kMaxExpectedCurrentAmps);
  // Use high-sensitivity ADC range (±40.96 mV) since the shunt drop at ~120 mA is ~1.8 mV.
  power_monitor.setADCRange(INA228_ADCRANGE_40_96MV); // 0: ±163.84mV, 1: ±40.96mV
  power_monitor.setCurrentConversionTime(kCurrentConversionTime);
  power_monitor.setVoltageConversionTime(kVoltageConversionTime);
  power_monitor.setTemperatureConversionTime(INA228_TIME_150_us);
  power_monitor.setAveragingCount(kAveragingCount);
  power_monitor.setMode(INA228_MODE_CONTINUOUS);
  power_monitor.resetAccumulators();
  power_monitor_ready = true;
  Serial.println("energy monitor: ready");
  delay(30);  // allow INA228 moving average to settle before sampling idle
  idle_baseline_power_mw = power_monitor.getPower_mW();
  return true;
}

float FlushEnergyWindow() {
  if (!power_monitor_ready) {
    return kInvalidTelemetryValue;
  }

  // Triggered conversion mode flushes the ADC pipeline and halts conversions
  // once the final averaged sample completes. Waiting kTriggeredFlushDelayMs
  // (≈28 ms conversion window) ensures the accumulators stop before reading.
  power_monitor.setMode(kTriggerStopMode);
  delay(kTriggeredFlushDelayMs);
  const float energy_total_j = power_monitor.readEnergy();
  power_monitor.resetAccumulators();
  power_monitor.setMode(INA228_MODE_CONTINUOUS);
  return energy_total_j;
}

void EmitPowerTelemetry(uint32_t sequence_id, float latency_s, int runs_completed,
                        float baseline_power_mw, float energy_total_j) {
  // Check if total energy measurement is valid (non-negative indicates successful reading)
  const bool energy_valid = (energy_total_j >= 0.0f);
  
  // Compute average energy consumed per inference (total energy divided by number of runs)
  const float energy_per_inference_j =
      (runs_completed > 0 && energy_valid) ? (energy_total_j / static_cast<float>(runs_completed))
                                           : kInvalidTelemetryValue;
  
  // Calculate average power in milliwatts: Power = Energy / Time, converted to mW
  // Note: per-inference energy is used here, so this is the average power per inference
  const float avg_power_per_inference_mw =
      (latency_s > 0.0f && energy_per_inference_j >= 0.0f)
          ? (energy_per_inference_j / latency_s) * 1000.0f
          : kInvalidTelemetryValue;
  
  // Read current bus voltage from the power monitor
  const float bus_voltage_v = power_monitor.getBusVoltage_V();
  
  // Calculate average current in milliamps using Ohm's law: Current = Power / Voltage
  const float avg_current_ma =
      (bus_voltage_v > 0.0f && avg_power_per_inference_mw >= 0.0f) ? (avg_power_per_inference_mw / bus_voltage_v)
                                                     : kInvalidTelemetryValue;

  Serial.print("inference seq: ");
  Serial.println(sequence_id);
  Serial.print("energy output (mJ): ");
  Serial.println(SanitizeFloat(energy_per_inference_j * 1000.0f), 6);
  Serial.print("avg power output (mW): ");
  Serial.println(SanitizeFloat(avg_power_per_inference_mw), 3);
  Serial.print("avg current output (mA): ");
  Serial.println(SanitizeFloat(avg_current_ma), 3);
  Serial.print("bus voltage output (V): ");
  Serial.println(SanitizeFloat(bus_voltage_v), 3);
  Serial.print("idle power baseline (mW): ");
  Serial.println(SanitizeFloat(baseline_power_mw), 3);
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

  // Serial.print("kTensorArenaSize: ");
  // Serial.println(kTensorArenaSize);

  Wire.begin();
  InitializePowerMonitor();

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
  if (power_monitor_ready) {
    power_monitor.resetAccumulators();
  }
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
    const float energy_total_j = FlushEnergyWindow();
    const float latency_s =
        (static_cast<float>(total_latency_us) / runs_completed) / 1000000.0f;
    ++inference_seq;
    EmitPowerTelemetry(inference_seq, latency_s, runs_completed, idle_baseline_power_mw, energy_total_j);
    // Emit the latency line expected by hardware_utils.py.
    Serial.print("timer output: ");
    Serial.println(latency_s, 6);
  }
}

void loop() {
  // Nothing else to do; keep the MCU alive for serial viewing.
  delay(1000);
}
