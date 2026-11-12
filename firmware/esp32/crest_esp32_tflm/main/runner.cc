// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "fixtures.h"
#include "model_data.h"
#include "crest_manifest.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef CREST_TENSOR_ARENA_BYTES
#define CREST_TENSOR_ARENA_BYTES (2048 * 1024)
#endif
#ifndef CREST_INFERENCE_RUNS
#define CREST_INFERENCE_RUNS 10
#endif
#ifndef CREST_WARMUP_RUNS
#define CREST_WARMUP_RUNS 1
#endif
#ifndef CREST_AGGREGATE_MEDIAN
#define CREST_AGGREGATE_MEDIAN 1
#endif
#ifndef CREST_HARNESS_TRIGGER_PIN
#define CREST_HARNESS_TRIGGER_PIN 21
#endif
#ifndef CREST_HARNESS_ARM_PIN
#define CREST_HARNESS_ARM_PIN 47
#endif
#ifndef CREST_DUT_ARM_HOLD_MS
#define CREST_DUT_ARM_HOLD_MS 600
#endif

namespace {
bool ReadStart() {
  char line[16] = {};
  size_t used = 0;
  while (used + 1 < sizeof(line)) {
    uint8_t byte = 0;
    const int count = uart_read_bytes(UART_NUM_0, &byte, 1, pdMS_TO_TICKS(1000));
    if (count <= 0) continue;
    if (byte == '\n' || byte == '\r') {
      if (used == 0) continue;
      line[used] = 0;
      return std::strcmp(line, "START") == 0;
    }
    line[used++] = static_cast<char>(byte);
  }
  return false;
}

[[noreturn]] void ReportFatalOnStart(const char *message) {
  while (true) {
    puts("DUT READY");
    if (ReadStart()) puts(message);
  }
}

long long Aggregate(long long *values, size_t count) {
#if CREST_AGGREGATE_MEDIAN
  std::sort(values, values + count);
  if (count % 2) return values[count / 2];
  return (values[count / 2 - 1] + values[count / 2]) / 2;
#else
  long long total = 0;
  for (size_t i = 0; i < count; ++i) total += values[i];
  return total / static_cast<long long>(count);
#endif
}
}  // namespace

extern "C" void app_main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  uart_driver_install(UART_NUM_0, 8192, 0, 0, nullptr, 0);
  gpio_config_t outputs = {};
  outputs.pin_bit_mask = (1ULL << CREST_HARNESS_TRIGGER_PIN) | (1ULL << CREST_HARNESS_ARM_PIN);
  outputs.mode = GPIO_MODE_OUTPUT;
  if (gpio_config(&outputs) != ESP_OK) { puts("ML_ERROR gpio"); return; }
  gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_TRIGGER_PIN), 0);
  gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_ARM_PIN), 1);

  auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
      16, CREST_TENSOR_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!arena) ReportFatalOnStart("failed to allocate tensor arena");
  const tflite::Model *model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) ReportFatalOnStart("ML_ERROR schema");
  tflite::MicroMutableOpResolver<17> ops;
  ops.AddAdd(); ops.AddAveragePool2D(); ops.AddConv2D(); ops.AddDepthwiseConv2D();
  ops.AddMaxPool2D(); ops.AddReshape(); ops.AddSoftmax(); ops.AddMul();
  ops.AddLogistic(); ops.AddFullyConnected(); ops.AddMean(); ops.AddReduceMax();
  ops.AddConcatenation(); ops.AddQuantize(); ops.AddShape();
  ops.AddStridedSlice(); ops.AddPack();
  tflite::MicroInterpreter interpreter(model, ops, arena, CREST_TENSOR_ARENA_BYTES);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ReportFatalOnStart("failed to allocate tensor arena: AllocateTensors failed");
  }
  if (interpreter.input(0)->type != kTfLiteInt8 ||
      interpreter.input(0)->bytes != kCrestFixtureBytes) {
    puts("ML_ERROR input_tensor"); return;
  }
  printf("ML_READY model_sha256=%s fixture_sha256=%s arena_used=%u arena_reserved=%u fixtures=%u warmups=%u repetitions=%u aggregation=%s\n",
         kCrestModelSha256, kCrestFixtureSha256,
         static_cast<unsigned>(interpreter.arena_used_bytes()),
         static_cast<unsigned>(CREST_TENSOR_ARENA_BYTES), kCrestFixtureCount,
         static_cast<unsigned>(CREST_WARMUP_RUNS), static_cast<unsigned>(CREST_INFERENCE_RUNS),
         CREST_AGGREGATE_MEDIAN ? "median" : "mean");

  while (true) {
    puts("DUT READY");
    if (!ReadStart()) continue;
    printf("ML_READY model_sha256=%s fixture_sha256=%s arena_used=%u arena_reserved=%u fixtures=%u warmups=%u repetitions=%u aggregation=%s\n",
           kCrestModelSha256, kCrestFixtureSha256,
           static_cast<unsigned>(interpreter.arena_used_bytes()),
           static_cast<unsigned>(CREST_TENSOR_ARENA_BYTES), kCrestFixtureCount,
           static_cast<unsigned>(CREST_WARMUP_RUNS), static_cast<unsigned>(CREST_INFERENCE_RUNS),
           CREST_AGGREGATE_MEDIAN ? "median" : "mean");
    gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_ARM_PIN), 0);
    vTaskDelay(pdMS_TO_TICKS(CREST_DUT_ARM_HOLD_MS));
    gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_TRIGGER_PIN), 1);
    constexpr size_t kSamples = kCrestFixtureCount * CREST_INFERENCE_RUNS;
    auto *timings = static_cast<long long *>(heap_caps_malloc(
        kSamples * sizeof(long long), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!timings) {
      gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_TRIGGER_PIN), 0);
      gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_ARM_PIN), 1);
      puts("ML_ERROR timing_allocation"); continue;
    }
    size_t sample = 0;
    bool failed = false;
    for (size_t fixture = 0; fixture < kCrestFixtureCount && !failed; ++fixture) {
      for (size_t warmup = 0; warmup < CREST_WARMUP_RUNS; ++warmup) {
        std::memcpy(interpreter.input(0)->data.int8, kCrestFixtureInputs[fixture], kCrestFixtureBytes);
        if (interpreter.Invoke() != kTfLiteOk) { failed = true; break; }
        vTaskDelay(pdMS_TO_TICKS(20));
      }
      for (size_t repetition = 0; repetition < CREST_INFERENCE_RUNS && !failed; ++repetition) {
        std::memcpy(interpreter.input(0)->data.int8, kCrestFixtureInputs[fixture], kCrestFixtureBytes);
        const int64_t start = esp_timer_get_time();
        const TfLiteStatus result = interpreter.Invoke();
        timings[sample] = esp_timer_get_time() - start;
        if (result != kTfLiteOk) { failed = true; break; }
        printf("ML_SAMPLE fixture=%u id=%s repetition=%u invoke_us=%lld\n",
               static_cast<unsigned>(fixture), kCrestFixtureIds[fixture],
               static_cast<unsigned>(repetition), timings[sample]);
        ++sample;
        vTaskDelay(pdMS_TO_TICKS(20));
      }
      if (!failed) {
        auto *output = interpreter.output(0);
        if (output->type != kTfLiteInt8) { failed = true; }
        else {
          printf("ML_OUTPUT fixture=%u id=%s values=", static_cast<unsigned>(fixture), kCrestFixtureIds[fixture]);
          for (size_t index = 0; index < output->bytes; ++index) {
            printf("%s%d", index ? "," : "", static_cast<int>(output->data.int8[index]));
          }
          puts("");
        }
      }
    }
    gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_TRIGGER_PIN), 0);
    gpio_set_level(static_cast<gpio_num_t>(CREST_HARNESS_ARM_PIN), 1);
    if (failed || sample != kSamples) { heap_caps_free(timings); puts("ML_ERROR invoke"); continue; }
    const long long aggregate_us = Aggregate(timings, sample);
    heap_caps_free(timings);
    printf("MEASUREMENT_VALID model_sha256=%s fixture_sha256=%s samples=%u\n",
           kCrestModelSha256, kCrestFixtureSha256, static_cast<unsigned>(sample));
    printf("runs: %u\n", static_cast<unsigned>(sample));
    printf("timer output: %.9f\n", static_cast<double>(aggregate_us) / 1000000.0);
  }
}
