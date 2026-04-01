#pragma once

#include <Arduino.h>
#include <stdint.h>

#if defined(DWT) && defined(CoreDebug) && defined(CoreDebug_DEMCR_TRCENA_Msk) && defined(DWT_CTRL_CYCCNTENA_Msk)
#define TINYODOM_HAS_DWT_CYCLE_COUNTER 1
#else
#define TINYODOM_HAS_DWT_CYCLE_COUNTER 0
#endif

namespace tinyodom_clock {

inline void ConfigureCycleCounter() {
#if TINYODOM_HAS_DWT_CYCLE_COUNTER
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
#endif
}

inline uint32_t ReadCycleCounterRaw() {
#if TINYODOM_HAS_DWT_CYCLE_COUNTER
  return DWT->CYCCNT;
#else
  return 0u;
#endif
}

inline float ReadClockHz() {
#if defined(ARDUINO_ARCH_MBED)
  const float runtime_clock_hz = static_cast<float>(SystemCoreClock);
  if (runtime_clock_hz > 0.0f) {
    return runtime_clock_hz;
  }
#endif
#if defined(F_CPU)
  const float compile_clock_hz = static_cast<float>(F_CPU);
  if (compile_clock_hz > 0.0f) {
    return compile_clock_hz;
  }
#endif
  return -1.0f;
}

inline float ComputeCyclesPerInference(
    uint32_t start_cycles,
    uint32_t end_cycles,
    int runs_completed,
    uint32_t total_latency_us,
    float clock_hz) {
#if TINYODOM_HAS_DWT_CYCLE_COUNTER
  if (runs_completed <= 0) {
    return -1.0f;
  }
  if (clock_hz > 0.0f) {
    constexpr double kDwtCycleCounterRange =
        static_cast<double>(UINT32_MAX) + 1.0;
    const double total_cycles_estimate =
        (static_cast<double>(total_latency_us) * static_cast<double>(clock_hz)) /
        1000000.0;
    // Unsigned subtraction is only unambiguous while the measurement window
    // stays below the full 32-bit counter range.
    if (total_cycles_estimate >= kDwtCycleCounterRange) {
      return -1.0f;
    }
  }
  const uint32_t total_cycles = end_cycles - start_cycles;
  return static_cast<float>(total_cycles) / static_cast<float>(runs_completed);
#else
  (void)start_cycles;
  (void)end_cycles;
  (void)runs_completed;
  (void)total_latency_us;
  (void)clock_hz;
  return -1.0f;
#endif
}

inline void EmitClockTelemetry(float clock_hz, float dwt_cycles_per_inference) {
  Serial.print("clock hz output: ");
  if (clock_hz >= 0.0f) {
    Serial.println(clock_hz, 0);
  } else {
    Serial.println(-1);
  }

  Serial.print("dwt cycles per inference output: ");
  if (dwt_cycles_per_inference >= 0.0f) {
    Serial.println(dwt_cycles_per_inference, 3);
  } else {
    Serial.println(-1);
  }
}

}  // namespace tinyodom_clock
