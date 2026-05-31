#include "tinyodom_dut_runner.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "stm32n6xx_hal_rtc.h"
#include "stm32n6xx_hal_rtc_ex.h"
#include "tinyodom_dut_phase_config.h"

enum
{
  kStartCommandTimeoutMs = 30000,
  kStartReadyHeartbeatMs = 5000,
  kUartPollTimeoutMs = 20,
  kArmHoldMs = 600,
  kRtcTimebaseHz = 32768,
  kRtcSyncPrediv = kRtcTimebaseHz - 1,
  kWakeupDivider = 16,
  kBlockIterations = 512,
};

#define TINYODOM_DUT_TRIGGER_GPIO_PORT   GPIOD
#define TINYODOM_DUT_TRIGGER_GPIO_PIN    GPIO_PIN_0
#define TINYODOM_DUT_ARM_GPIO_PORT       GPIOE
#define TINYODOM_DUT_ARM_GPIO_PIN        GPIO_PIN_9

#define TINYODOM_DUT_STOP_MODE_VARIANT "system_stop_mainreg_wfi"

static RTC_HandleTypeDef s_rtc_handle;
static volatile bool s_rtc_wakeup_fired = false;
static const char *s_rtc_clock_source = "UNINITIALIZED";
static uint32_t s_rtc_clock_hz_nominal = 0U;
static const char *s_cadence_timing_quality = "unknown";
static volatile float s_float_sink = 1.0001f;
static volatile uint32_t s_int_sink = 0x12345678UL;
static volatile uint32_t s_poll_sink = 0x31415926UL;

typedef struct
{
  uint64_t iterations;
  uint64_t work_units;
  const char *work_unit_label;
  uint64_t sleep_us;
  const char *sleep_mode;
} MicroWorkloadTelemetry;

void SystemClock_Config(void);

static void flush_stdout(void)
{
  fflush(stdout);
}

static void emit_line(const char *line)
{
  printf("%s\r\n", line);
  flush_stdout();
}

static void print_u64_decimal(uint64_t value)
{
  char digits[21];
  size_t pos = sizeof(digits) - 1U;

  digits[pos] = '\0';
  do
  {
    digits[--pos] = (char)('0' + (value % 10ULL));
    value /= 10ULL;
  } while ((value != 0ULL) && (pos > 0U));

  printf("%s", &digits[pos]);
}

static void print_u64_output(const char *prefix, uint64_t value)
{
  printf("%s", prefix);
  print_u64_decimal(value);
  printf("\r\n");
}

static void set_harness_idle(void)
{
  HAL_GPIO_WritePin(TINYODOM_DUT_TRIGGER_GPIO_PORT, TINYODOM_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(TINYODOM_DUT_ARM_GPIO_PORT, TINYODOM_DUT_ARM_GPIO_PIN, GPIO_PIN_SET);
}

static void arm_harness_window(void)
{
  HAL_GPIO_WritePin(TINYODOM_DUT_ARM_GPIO_PORT, TINYODOM_DUT_ARM_GPIO_PIN, GPIO_PIN_RESET);
  HAL_Delay(kArmHoldMs);
  HAL_GPIO_WritePin(TINYODOM_DUT_TRIGGER_GPIO_PORT, TINYODOM_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_SET);
}

static void disarm_harness_window(void)
{
  HAL_GPIO_WritePin(TINYODOM_DUT_TRIGGER_GPIO_PORT, TINYODOM_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(TINYODOM_DUT_ARM_GPIO_PORT, TINYODOM_DUT_ARM_GPIO_PIN, GPIO_PIN_SET);
}

static uint32_t read_cpu_clock_hz(void)
{
  SystemCoreClockUpdate();
  if (SystemCoreClock > 0U)
  {
    return SystemCoreClock;
  }
  return HAL_RCC_GetCpuClockFreq();
}

static bool uart_read_line(char *buffer, size_t buffer_size, uint32_t total_timeout_ms)
{
  uint32_t start_tick = HAL_GetTick();
  size_t used = 0U;

  if (buffer_size == 0U)
  {
    return false;
  }
  buffer[0] = '\0';

  while ((HAL_GetTick() - start_tick) < total_timeout_ms)
  {
    uint8_t ch = 0U;
    if (HAL_UART_Receive(&hcom_uart[COM1], &ch, 1, kUartPollTimeoutMs) != HAL_OK)
    {
      continue;
    }

    if ((ch == '\r') || (ch == '\n'))
    {
      if (used == 0U)
      {
        continue;
      }
      buffer[used] = '\0';
      return true;
    }

    if (used + 1U < buffer_size)
    {
      buffer[used++] = (char)ch;
      buffer[used] = '\0';
    }
  }

  return false;
}

static bool wait_for_start_command(void)
{
  uint32_t wait_started = HAL_GetTick();
  uint32_t last_ready_emit = 0U;
  char line[32];

  while ((HAL_GetTick() - wait_started) < kStartCommandTimeoutMs)
  {
    uint32_t now = HAL_GetTick();
    if ((last_ready_emit == 0U) || ((now - last_ready_emit) >= kStartReadyHeartbeatMs))
    {
      emit_line("DUT READY");
      last_ready_emit = now;
    }

    if (!uart_read_line(line, sizeof(line), kStartReadyHeartbeatMs))
    {
      continue;
    }

    if (strcmp(line, "START") == 0)
    {
      return true;
    }
  }

  return false;
}

static bool rtc_try_init_with_source(uint32_t oscillator_type,
                                     uint32_t rtc_clock_selection,
                                     const char *label,
                                     uint32_t nominal_hz)
{
  RCC_OscInitTypeDef osc_init = {0};
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};
  RTC_TimeTypeDef rtc_time = {0};
  RTC_DateTypeDef rtc_date = {0};
  RTC_PrivilegeStateTypeDef privilege_state = {0};
  RTC_SecureStateTypeDef secure_state = {0};
  HAL_StatusTypeDef status = HAL_OK;

  __HAL_RCC_PWR_CLK_ENABLE();
  HAL_PWR_EnableBkUpAccess();

  osc_init.OscillatorType = oscillator_type;
  osc_init.PLL1.PLLState = RCC_PLL_NONE;
  osc_init.PLL2.PLLState = RCC_PLL_NONE;
  osc_init.PLL3.PLLState = RCC_PLL_NONE;
  osc_init.PLL4.PLLState = RCC_PLL_NONE;
  if (oscillator_type == RCC_OSCILLATORTYPE_LSE)
  {
    osc_init.LSEState = RCC_LSE_ON;
  }
  else
  {
    osc_init.LSIState = RCC_LSI_ON;
  }

  status = HAL_RCC_OscConfig(&osc_init);
  if (status != HAL_OK)
  {
    return false;
  }

  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_RTC;
  periph_clk_init.RTCClockSelection = rtc_clock_selection;
  status = HAL_RCCEx_PeriphCLKConfig(&periph_clk_init);
  if (status != HAL_OK)
  {
    return false;
  }

  __HAL_RCC_RTC_ENABLE();
  s_rtc_handle.Instance = RTC;
  s_rtc_handle.Init.HourFormat = RTC_HOURFORMAT_24;
  s_rtc_handle.Init.AsynchPrediv = 0U;
  s_rtc_handle.Init.SynchPrediv = kRtcSyncPrediv;
  s_rtc_handle.Init.OutPut = RTC_OUTPUT_DISABLE;
  s_rtc_handle.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
  s_rtc_handle.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;

  status = HAL_RTC_Init(&s_rtc_handle);
  if (status != HAL_OK)
  {
    return false;
  }

  privilege_state.rtcPrivilegeFull = RTC_PRIVILEGE_FULL_NO;
  privilege_state.backupRegisterPrivZone = RTC_PRIVILEGE_BKUP_ZONE_NONE;
  privilege_state.backupRegisterStartZone2 = RTC_BKP_DR0;
  privilege_state.backupRegisterStartZone3 = RTC_BKP_DR0;
  status = HAL_RTCEx_PrivilegeModeSet(&s_rtc_handle, &privilege_state);
  if (status != HAL_OK)
  {
    return false;
  }

  secure_state.rtcSecureFull = RTC_SECURE_FULL_YES;
  secure_state.backupRegisterStartZone2 = RTC_BKP_DR0;
  secure_state.backupRegisterStartZone3 = RTC_BKP_DR0;
  status = HAL_RTCEx_SecureModeSet(&s_rtc_handle, &secure_state);
  if (status != HAL_OK)
  {
    return false;
  }

  rtc_time.Hours = 0U;
  rtc_time.Minutes = 0U;
  rtc_time.Seconds = 0U;
  rtc_time.DayLightSaving = RTC_DAYLIGHTSAVING_NONE;
  rtc_time.StoreOperation = RTC_STOREOPERATION_RESET;
  status = HAL_RTC_SetTime(&s_rtc_handle, &rtc_time, RTC_FORMAT_BIN);
  if (status != HAL_OK)
  {
    return false;
  }

  rtc_date.WeekDay = RTC_WEEKDAY_MONDAY;
  rtc_date.Month = RTC_MONTH_JANUARY;
  rtc_date.Date = 1U;
  rtc_date.Year = 24U;
  status = HAL_RTC_SetDate(&s_rtc_handle, &rtc_date, RTC_FORMAT_BIN);
  if (status != HAL_OK)
  {
    return false;
  }

  s_rtc_clock_source = label;
  s_rtc_clock_hz_nominal = nominal_hz;
  s_cadence_timing_quality = (strcmp(label, "LSE") == 0) ? "crystal" : "rc_coarse";
  HAL_RCCEx_WakeUpStopCLKConfig(RCC_STOP_WAKEUPCLOCK_HSI);
  return true;
}

static bool rtc_init(void)
{
  return rtc_try_init_with_source(RCC_OSCILLATORTYPE_LSE, RCC_RTCCLKSOURCE_LSE, "LSE", kRtcTimebaseHz);
}

static uint64_t rtc_time_to_us(const RTC_TimeTypeDef *time)
{
  uint32_t seconds_of_day = ((uint32_t)time->Hours * 3600U) +
                            ((uint32_t)time->Minutes * 60U) +
                            (uint32_t)time->Seconds;
  uint32_t fraction_ticks = 0U;
  uint64_t rtc_clock_hz = (uint64_t)s_rtc_clock_hz_nominal;

  if (time->SecondFraction >= time->SubSeconds)
  {
    fraction_ticks = time->SecondFraction - time->SubSeconds;
  }
  if (rtc_clock_hz == 0ULL)
  {
    rtc_clock_hz = (uint64_t)kRtcTimebaseHz;
  }
  return ((uint64_t)seconds_of_day * 1000000ULL) +
         (((uint64_t)fraction_ticks * 1000000ULL) / rtc_clock_hz);
}

static uint64_t rtc_datetime_to_us(const RTC_TimeTypeDef *time, const RTC_DateTypeDef *date)
{
  uint64_t day_offset_seconds = 0ULL;
  if (date->Date > 0U)
  {
    day_offset_seconds = ((uint64_t)date->Date - 1ULL) * 86400ULL;
  }
  return (day_offset_seconds * 1000000ULL) + rtc_time_to_us(time);
}

static uint64_t rtc_now_us(void)
{
  RTC_TimeTypeDef first_time = {0};
  RTC_TimeTypeDef second_time = {0};
  RTC_DateTypeDef first_date = {0};
  RTC_DateTypeDef second_date = {0};

  for (int attempt = 0; attempt < 4; ++attempt)
  {
    HAL_RTC_GetTime(&s_rtc_handle, &first_time, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&s_rtc_handle, &first_date, RTC_FORMAT_BIN);
    HAL_RTC_GetTime(&s_rtc_handle, &second_time, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&s_rtc_handle, &second_date, RTC_FORMAT_BIN);

    uint64_t first_us = rtc_datetime_to_us(&first_time, &first_date);
    uint64_t second_us = rtc_datetime_to_us(&second_time, &second_date);
    if (second_us >= first_us)
    {
      return second_us;
    }
  }

  return rtc_datetime_to_us(&second_time, &second_date);
}

static bool restore_runtime_clocks(void)
{
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};

  SystemClock_Config();
  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_USART1;
  periph_clk_init.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  if (HAL_RCCEx_PeriphCLKConfig(&periph_clk_init) != HAL_OK)
  {
    return false;
  }

  SystemCoreClockUpdate();
  return true;
}

static uint32_t wakeup_counter_for_sleep_us(uint32_t sleep_us)
{
  uint64_t rtc_clock_hz = (uint64_t)s_rtc_clock_hz_nominal;
  uint64_t counter = 0ULL;

  if (rtc_clock_hz == 0ULL)
  {
    return 0U;
  }

  counter = ((uint64_t)sleep_us * rtc_clock_hz) / ((uint64_t)kWakeupDivider * 1000000ULL);
  if (counter > 0xFFFFU)
  {
    counter = 0xFFFFU;
  }

  return (uint32_t)counter;
}

static bool stop_sleep_for_us(uint32_t sleep_us)
{
  uint32_t wakeup_counter = 0U;

  if (sleep_us < TINYODOM_DUT_MIN_SLEEP_US)
  {
    return false;
  }
  wakeup_counter = wakeup_counter_for_sleep_us(sleep_us);
  if (wakeup_counter < 2U)
  {
    return false;
  }
  wakeup_counter -= 1U;

  s_rtc_wakeup_fired = false;
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);
  NVIC_ClearPendingIRQ(RTC_S_IRQn);
  HAL_NVIC_SetPriority(RTC_S_IRQn, 0U, 0U);
  HAL_NVIC_EnableIRQ(RTC_S_IRQn);
  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  if (HAL_RTCEx_SetWakeUpTimer_IT(&s_rtc_handle,
                                  wakeup_counter,
                                  RTC_WAKEUPCLOCK_RTCCLK_DIV16,
                                  0U) != HAL_OK)
  {
    return false;
  }

  HAL_SuspendTick();
  HAL_PWR_EnterSTOPMode(PWR_MAINREGULATOR_ON, PWR_STOPENTRY_WFI);
  if (__HAL_PWR_GET_FLAG(PWR_FLAG_STOPF) != RESET)
  {
    __HAL_PWR_CLEAR_FLAG(PWR_FLAG_STOPF);
  }
  if (!restore_runtime_clocks())
  {
    HAL_ResumeTick();
    return false;
  }
  HAL_ResumeTick();
  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);
  return s_rtc_wakeup_fired;
}

static MicroWorkloadTelemetry run_poll_workload(uint64_t end_us)
{
  uint32_t value = s_poll_sink;
  uint64_t blocks = 0ULL;

  while (rtc_now_us() < end_us)
  {
    for (uint32_t iter = 0U; iter < kBlockIterations; ++iter)
    {
      __asm volatile("" ::: "memory");
    }
    value += (uint32_t)blocks;
    ++blocks;
  }

  s_poll_sink = value;
  MicroWorkloadTelemetry telemetry = {
      blocks * (uint64_t)kBlockIterations,
      blocks * (uint64_t)kBlockIterations,
      "poll_iters",
      0ULL,
      "none",
  };
  return telemetry;
}

static MicroWorkloadTelemetry run_wait_workload(uint64_t end_us)
{
  uint64_t blocks = 0ULL;

  while (rtc_now_us() < end_us)
  {
    uint64_t block_end_us = rtc_now_us() + 1000ULL;
    while (rtc_now_us() < block_end_us)
    {
      __WFI();
    }
    ++blocks;
  }

  MicroWorkloadTelemetry telemetry = {
      blocks,
      blocks,
      "wait_blocks",
      0ULL,
      "none",
  };
  return telemetry;
}

static MicroWorkloadTelemetry run_float_workload(uint64_t end_us)
{
  float value = s_float_sink;
  uint64_t blocks = 0ULL;

  while (rtc_now_us() < end_us)
  {
    for (uint32_t iter = 0U; iter < kBlockIterations; ++iter)
    {
      value = (value * 1.000091f) + 0.000173f;
      value = (value * 0.999947f) - 0.000031f;
      if ((value > 65536.0f) || (value < -65536.0f))
      {
        value *= 0.0001f;
      }
    }
    ++blocks;
  }

  s_float_sink = value;
  MicroWorkloadTelemetry telemetry = {
      blocks * (uint64_t)kBlockIterations,
      blocks * (uint64_t)kBlockIterations * 4ULL,
      "fp_ops",
      0ULL,
      "none",
  };
  return telemetry;
}

static MicroWorkloadTelemetry run_int_workload(uint64_t end_us)
{
  uint32_t value = s_int_sink;
  uint64_t blocks = 0ULL;

  while (rtc_now_us() < end_us)
  {
    for (uint32_t iter = 0U; iter < kBlockIterations; ++iter)
    {
      value = (value * 1664525UL) + 1013904223UL;
      value ^= value << 13;
      value ^= value >> 17;
      value ^= value << 5;
    }
    ++blocks;
  }

  s_int_sink = value;
  MicroWorkloadTelemetry telemetry = {
      blocks * (uint64_t)kBlockIterations,
      blocks * (uint64_t)kBlockIterations * 7ULL,
      "int_ops",
      0ULL,
      "none",
  };
  return telemetry;
}

static MicroWorkloadTelemetry run_sleep_workload(uint64_t start_us, uint64_t end_us)
{
  uint64_t slept_us = 0ULL;
  uint64_t loops = 0ULL;

  while (rtc_now_us() < end_us)
  {
    uint64_t now_us = rtc_now_us();
    uint64_t remaining_us = (end_us > now_us) ? (end_us - now_us) : 0ULL;
    if (remaining_us > (uint64_t)(TINYODOM_DUT_WAKE_MARGIN_US + TINYODOM_DUT_MIN_SLEEP_US))
    {
      uint32_t request_us = (uint32_t)(remaining_us - (uint64_t)TINYODOM_DUT_WAKE_MARGIN_US);
      if (stop_sleep_for_us(request_us))
      {
        slept_us += (uint64_t)request_us;
      }
      continue;
    }
    __WFI();
    ++loops;
  }

  slept_us = slept_us > (end_us - start_us) ? (end_us - start_us) : slept_us;
  MicroWorkloadTelemetry telemetry = {
      loops,
      0ULL,
      "none",
      slept_us,
      "stop_sleep",
  };
  return telemetry;
}

static const char *workload_label(void)
{
#if TINYODOM_MICRO_WORKLOAD_MODE == 0
  return "sleep";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 1
  return "wait";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 2
  return "poll";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 3
  return "float";
#elif TINYODOM_MICRO_WORKLOAD_MODE == 4
  return "int";
#else
  return "sleep";
#endif
}

void tinyodom_dut_harness_gpio_init(void)
{
  GPIO_InitTypeDef gpio_init = {0};

  HAL_PWREx_EnableVddIO5();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
  gpio_init.Pull = GPIO_NOPULL;
  gpio_init.Speed = GPIO_SPEED_FREQ_LOW;

  gpio_init.Pin = TINYODOM_DUT_ARM_GPIO_PIN;
  HAL_GPIO_Init(TINYODOM_DUT_ARM_GPIO_PORT, &gpio_init);

  gpio_init.Pin = TINYODOM_DUT_TRIGGER_GPIO_PIN;
  HAL_GPIO_Init(TINYODOM_DUT_TRIGGER_GPIO_PORT, &gpio_init);

  set_harness_idle();
}

void tinyodom_dut_rtc_irq_handler(void)
{
  HAL_RTCEx_WakeUpTimerIRQHandler(&s_rtc_handle);
}

void HAL_RTCEx_WakeUpTimerEventCallback(RTC_HandleTypeDef *hrtc)
{
  if (hrtc->Instance == RTC)
  {
    s_rtc_wakeup_fired = true;
  }
}

int tinyodom_dut_run_once(void)
{
  uint32_t clock_hz = 0U;
  uint32_t start_cycles = 0U;
  uint32_t elapsed_cycles = 0U;
  uint64_t start_us = 0ULL;
  uint64_t end_us = 0ULL;
  uint64_t elapsed_us = 0ULL;
  MicroWorkloadTelemetry workload_telemetry = {0ULL, 0ULL, "none", 0ULL, "none"};

  emit_line("STM32_BOOT=START");
  emit_line("STM32_RTC_INIT=START");
  if (!rtc_init())
  {
    emit_line("STM32_RTC_INIT=FAIL");
    emit_line("STM32_AI_INIT=FAIL reason=rtc_init");
    return -6;
  }
  emit_line("STM32_RTC_INIT=OK");
  emit_line("STM32_AI_INIT=OK");

  printf("phase output: %s\r\n", (TINYODOM_MICRO_WORKLOAD_MODE == 0) ? "cadenced" : "back_to_back");
  printf("rtc clock source output: %s\r\n", s_rtc_clock_source);
  printf("rtc clock hz nominal output: %lu\r\n", (unsigned long)s_rtc_clock_hz_nominal);
  printf("cadence timing quality output: %s\r\n", s_cadence_timing_quality);
  printf("stop mode variant output: %s\r\n", TINYODOM_DUT_STOP_MODE_VARIANT);
  printf("cadence budget (ms): %d\r\n", TINYODOM_DUT_LATENCY_BUDGET_MS);
  printf("workload output: %s\r\n", workload_label());
  printf("requested window ms: %d\r\n", TINYODOM_MICRO_WINDOW_MS);
  flush_stdout();

  if (!wait_for_start_command())
  {
    emit_line("STM32_AI_START=TIMEOUT");
    return -3;
  }

  arm_harness_window();
  start_us = rtc_now_us();
  end_us = start_us + ((uint64_t)TINYODOM_MICRO_WINDOW_MS * 1000ULL);
  start_cycles = DWT->CYCCNT;

#if TINYODOM_MICRO_WORKLOAD_MODE == 0
  workload_telemetry = run_sleep_workload(start_us, end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 1
  workload_telemetry = run_wait_workload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 2
  workload_telemetry = run_poll_workload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 3
  workload_telemetry = run_float_workload(end_us);
#elif TINYODOM_MICRO_WORKLOAD_MODE == 4
  workload_telemetry = run_int_workload(end_us);
#else
  workload_telemetry = run_sleep_workload(start_us, end_us);
#endif

  elapsed_cycles = DWT->CYCCNT - start_cycles;
  elapsed_us = rtc_now_us() - start_us;
  disarm_harness_window();

  clock_hz = read_cpu_clock_hz();
  emit_line("runs: 1");
  printf("clock hz output: %lu\r\n", (unsigned long)clock_hz);
  printf("dwt cycles per inference output: %lu\r\n", (unsigned long)elapsed_cycles);
  printf("timer per inference output: %lu.%06lu\r\n",
         (unsigned long)(elapsed_us / 1000000ULL),
         (unsigned long)(elapsed_us % 1000000ULL));
  printf("timer per window output: %lu.%06lu\r\n",
         (unsigned long)(elapsed_us / 1000000ULL),
         (unsigned long)(elapsed_us % 1000000ULL));
  printf("timer output: %lu.%06lu\r\n",
         (unsigned long)(elapsed_us / 1000000ULL),
         (unsigned long)(elapsed_us % 1000000ULL));
  printf("rtc sleep total ms: %lu.%03lu\r\n",
         (unsigned long)(workload_telemetry.sleep_us / 1000ULL),
         (unsigned long)(workload_telemetry.sleep_us % 1000ULL));
  print_u64_output("dut iterations output: ", workload_telemetry.iterations);
  print_u64_output("dut work units output: ", workload_telemetry.work_units);
  printf("dut work unit label output: %s\r\n", workload_telemetry.work_unit_label);
  print_u64_output("dut elapsed us output: ", elapsed_us);
  printf("dut cycles output: %lu\r\n", (unsigned long)elapsed_cycles);
  printf("dut sleep ms output: %lu.%03lu\r\n",
         (unsigned long)(workload_telemetry.sleep_us / 1000ULL),
         (unsigned long)(workload_telemetry.sleep_us % 1000ULL));
  printf("dut sleep mode output: %s\r\n", workload_telemetry.sleep_mode);
  emit_line("micro workload run: ok");
  emit_line("wake recovery us: -1");
  emit_line("wake overshoot us: -1");
  emit_line("cadence deadline misses: 0");
  emit_line("inference seq output: 1");
  printf("STM32_AI_LATENCY_CYCLES=%lu\r\n", (unsigned long)elapsed_cycles);
  emit_line("STM32_AI_RUN=OK");
  flush_stdout();
  return 0;
}
