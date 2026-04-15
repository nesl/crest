#include "toy_ai_runner.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "network.h"       /* generated: AI_NETWORK_* macros, ai_network_* API */
#include "network_data.h"  /* generated: AI_NETWORK_DATA_ACTIVATIONS_SIZE, weight storage */
#include "toy_ai_phase_config.h"
#include "stm32n6xx_nucleo_xspi.h"

/* Activations pool — the ST Edge AI runtime carves all intermediate tensor
 * buffers out of this block at init time.  The input and output tensors for
 * this model also live inside it (their .data pointers are set by
 * ai_network_create_and_init to offsets within this array).
 *
 * Size comes from the generated header; for the primary TinyODOM model it is
 * 38,200 bytes.
 *
 * Alignment: AI_ALIGNED(4) satisfies AI_NETWORK_ACTIVATIONS_ALIGNMENT for
 * this model.  If a future model raises that requirement, increase accordingly.
 *
 * IMPORTANT — linker script stack size: the blink FSBL default of 0x400 (1 kB)
 * is not enough for the inference call tree.  This project's linker script
 * must use at least 0x4000 (16 kB), matching CM55_Validation.  See
 * stm32_phase0_cpu_first_findings.md for the full diagnosis. */
AI_ALIGNED(4)
static ai_u8 s_network_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

enum
{
  kStartCommandTimeoutMs = 30000,
  kStartReadyHeartbeatMs = 5000,
  kUartPollTimeoutMs = 20,
  kArmHoldMs = 600,
};

static const uintptr_t kExternalWeightsStart = 0x70000000UL;
static const uintptr_t kExternalWeightsEnd = 0x80000000UL;

#define TOY_AI_TRIGGER_GPIO_PORT   GPIOD
#define TOY_AI_TRIGGER_GPIO_PIN    GPIO_PIN_0
#define TOY_AI_ARM_GPIO_PORT       GPIOE
#define TOY_AI_ARM_GPIO_PIN        GPIO_PIN_9

static void flush_stdout(void)
{
  fflush(stdout);
}

static void emit_line(const char *line)
{
  printf("%s\r\n", line);
  flush_stdout();
}

static void emit_cpu_clock_telemetry(uint32_t cpu_clock_hz)
{
  printf("clock hz output: %lu\r\n", (unsigned long)cpu_clock_hz);
  flush_stdout();
}

static uintptr_t generated_weights_address(void)
{
  return (uintptr_t)ai_network_data_weights_get();
}

static bool weights_use_external_flash(void)
{
  uintptr_t weights_addr = generated_weights_address();
  return (weights_addr >= kExternalWeightsStart) && (weights_addr < kExternalWeightsEnd);
}

static bool configure_external_weight_xspi2_clocking(void)
{
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};

  /* Mirror the Nucleo XIP template's XSPI2 clock-source selection for the
   * toy external-weight path. The BSP driver configures pins and the NOR
   * protocol, but it does not select the XSPI2 kernel clock or enable XSPIM. */
  __HAL_RCC_PWR_CLK_ENABLE();
  HAL_PWREx_EnableVddIO3();
  HAL_PWREx_ConfigVddIORange(PWR_VDDIO3, PWR_VDDIO_RANGE_1V8);

  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
  periph_clk_init.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
  periph_clk_init.ICSelection[RCC_IC3].ClockSelection = RCC_ICCLKSOURCE_PLL1;
  periph_clk_init.ICSelection[RCC_IC3].ClockDivider = 24;

  if (HAL_RCCEx_PeriphCLKConfig(&periph_clk_init) != HAL_OK)
  {
    return false;
  }

  __HAL_RCC_XSPIM_CLK_ENABLE();
  return true;
}

static bool ensure_external_weight_mapping_ready(void)
{
  BSP_XSPI_NOR_Init_t xspi_init = {0};
  uintptr_t weights_addr = generated_weights_address();
  volatile const uint32_t *probe_word = (const uint32_t *)weights_addr;
  uint32_t first_word = 0U;
  int32_t status = BSP_ERROR_NONE;

  if (!weights_use_external_flash())
  {
    return true;
  }

  xspi_init.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
  xspi_init.TransferRate = BSP_XSPI_NOR_DTR_TRANSFER;

  if (!configure_external_weight_xspi2_clocking())
  {
    printf("STM32_XSPI_INIT=FAIL stage=clock_config status=%d addr=0x%08lx\r\n",
           HAL_ERROR,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  status = BSP_XSPI_NOR_Init(0U, &xspi_init);
  if (status != BSP_ERROR_NONE)
  {
    printf("STM32_XSPI_INIT=FAIL stage=init status=%ld addr=0x%08lx\r\n",
           (long)status,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  status = BSP_XSPI_NOR_EnableMemoryMappedMode(0U);
  if (status != BSP_ERROR_NONE)
  {
    printf("STM32_XSPI_INIT=FAIL stage=memory_mapped status=%ld addr=0x%08lx\r\n",
           (long)status,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  /* External-flash weights are only valid once XSPI2 memory-mapped reads are
   * live. This probe read proves the generated default weight address is
   * CPU-readable before the AI runtime dereferences it. */
  __DSB();
  __ISB();
  first_word = *probe_word;
  printf("STM32_XSPI_INIT=OK addr=0x%08lx word0=0x%08lx\r\n",
         (unsigned long)weights_addr,
         (unsigned long)first_word);
  flush_stdout();
  return true;
}

static void emit_cycle_telemetry(uint64_t cycles_per_inference_milli, bool valid)
{
  if (valid)
  {
    unsigned long whole = (unsigned long)(cycles_per_inference_milli / 1000ULL);
    unsigned long frac = (unsigned long)(cycles_per_inference_milli % 1000ULL);
    printf("dwt cycles per inference output: %lu.%03lu\r\n", whole, frac);
  }
  else
  {
    emit_line("dwt cycles per inference output: -1");
    return;
  }
  flush_stdout();
}

static void emit_timer_telemetry(uint64_t seconds_per_inference_scaled_1e6, bool valid)
{
  if (valid)
  {
    unsigned long whole = (unsigned long)(seconds_per_inference_scaled_1e6 / 1000000ULL);
    unsigned long frac = (unsigned long)(seconds_per_inference_scaled_1e6 % 1000000ULL);
    printf("timer output: %lu.%06lu\r\n", whole, frac);
  }
  else
  {
    emit_line("timer output: -1");
    return;
  }
  flush_stdout();
}

static void set_harness_idle(void)
{
  HAL_GPIO_WritePin(TOY_AI_TRIGGER_GPIO_PORT, TOY_AI_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(TOY_AI_ARM_GPIO_PORT, TOY_AI_ARM_GPIO_PIN, GPIO_PIN_SET);
}

static void arm_harness_window(void)
{
  HAL_GPIO_WritePin(TOY_AI_ARM_GPIO_PORT, TOY_AI_ARM_GPIO_PIN, GPIO_PIN_RESET);
  HAL_Delay(kArmHoldMs);
  HAL_GPIO_WritePin(TOY_AI_TRIGGER_GPIO_PORT, TOY_AI_TRIGGER_GPIO_PIN, GPIO_PIN_SET);
}

static void disarm_harness_window(void)
{
  HAL_GPIO_WritePin(TOY_AI_TRIGGER_GPIO_PORT, TOY_AI_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(TOY_AI_ARM_GPIO_PORT, TOY_AI_ARM_GPIO_PIN, GPIO_PIN_SET);
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
  size_t used = 0;

  if (buffer_size == 0U)
  {
    return false;
  }
  buffer[0] = '\0';

  while ((HAL_GetTick() - start_tick) < total_timeout_ms)
  {
    uint8_t ch = 0;
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

/* Zero-fill the first (and only) input tensor.
 * The input .data pointer is already set to an offset inside s_network_activations
 * by ai_network_create_and_init, so this write goes directly into the
 * activations pool.  For Phase 0 we just want a known-good input; real
 * inference would copy sensor data here instead. */
static void fill_input_buffer(ai_buffer *input)
{
  memset((void *)input[0].data, 0, AI_NETWORK_IN_1_SIZE_BYTES);
}

void toy_ai_harness_gpio_init(void)
{
  GPIO_InitTypeDef gpio_init = {0};

  HAL_PWREx_EnableVddIO5();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
  gpio_init.Pull = GPIO_NOPULL;
  gpio_init.Speed = GPIO_SPEED_FREQ_LOW;

  gpio_init.Pin = TOY_AI_ARM_GPIO_PIN;
  HAL_GPIO_Init(TOY_AI_ARM_GPIO_PORT, &gpio_init);

  gpio_init.Pin = TOY_AI_TRIGGER_GPIO_PIN;
  HAL_GPIO_Init(TOY_AI_TRIGGER_GPIO_PORT, &gpio_init);

  set_harness_idle();
}

int toy_ai_run_once(void)
{
  ai_handle network = AI_HANDLE_NULL;
  uint32_t cpu_clock_hz = 0U;

  if (!ensure_external_weight_mapping_ready())
  {
    return -5;
  }

  /* Pass the activations pool to the runtime. NULL for weights tells the
   * legacy API to use the generated default weight mapping, which may resolve
   * either to an embedded array or to an external flash address depending on
   * the selected generation mode. */
  const ai_handle activations[] = {s_network_activations};
  ai_error err = ai_network_create_and_init(&network, activations, NULL);
  if (err.type != AI_ERROR_NONE)
  {
    printf("STM32_AI_INIT=FAIL type=%d code=%d\r\n", err.type, err.code);
    flush_stdout();
    return -1;
  }

  printf("STM32_AI_INIT=OK\r\n");
  flush_stdout();  /* newlib-nano buffers stdout; flush explicitly so the token
                    * is visible over UART before inference begins */

  /* Get pointers to the network's input and output buffer descriptors.
   * Both are backed by regions inside s_network_activations — their .data
   * fields were set during create_and_init and must not be reassigned. */
  ai_buffer *input  = ai_network_inputs_get(network, NULL);
  ai_buffer *output = ai_network_outputs_get(network, NULL);

  cpu_clock_hz = read_cpu_clock_hz();

  /* Warm-up run: drives the runtime through its full execution path once
   * before measurement.  This amortises any one-time setup cost (cache
   * warming, branch predictor state) so the timed run reflects steady-state
   * latency rather than cold-start behaviour. */
  fill_input_buffer(input);
  ai_i32 batch = ai_network_run(network, input, output);
  if (batch != 1)
  {
    printf("STM32_AI_RUN=FAIL batch=%ld\r\n", (long)batch);
    flush_stdout();
    ai_network_destroy(network);
    return -2;
  }

  if (!wait_for_start_command())
  {
    emit_line("STM32_AI_START=TIMEOUT");
    ai_network_destroy(network);
    return -3;
  }

  /* Timed run: DWT->CYCCNT is a free-running CPU cycle counter enabled in
   * DWT_Init() (main.c). Capture before and after the full configured window,
   * then normalize the result per inference. */
  fill_input_buffer(input);
  arm_harness_window();
  uint32_t start_cycles = DWT->CYCCNT;
  int runs_completed = 0;
  for (int run_idx = 0; run_idx < TOY_AI_MEASURED_RUNS; ++run_idx)
  {
    batch = ai_network_run(network, input, output);
    if (batch != 1)
    {
      disarm_harness_window();
      printf("STM32_AI_RUN=FAIL batch=%ld run=%d\r\n", (long)batch, run_idx + 1);
      flush_stdout();
      ai_network_destroy(network);
      return -4;
    }
    runs_completed++;
  }
  uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;
  disarm_harness_window();

  bool have_cycles = false;
  bool have_seconds = false;
  uint32_t cycles_per_inference_int = 0U;
  uint64_t cycles_per_inference_milli = 0ULL;
  uint64_t seconds_per_inference_scaled_1e6 = 0ULL;
  if (runs_completed > 0)
  {
    have_cycles = true;
    cycles_per_inference_int = elapsed_cycles / (uint32_t)runs_completed;
    cycles_per_inference_milli = ((uint64_t)elapsed_cycles * 1000ULL) / (uint64_t)runs_completed;
  }
  if (have_cycles && (cpu_clock_hz > 0U))
  {
    have_seconds = true;
    seconds_per_inference_scaled_1e6 =
        ((uint64_t)elapsed_cycles * 1000000ULL) /
        ((uint64_t)cpu_clock_hz * (uint64_t)runs_completed);
  }

  printf("runs: %d\r\n", runs_completed);
  emit_cpu_clock_telemetry(cpu_clock_hz);
  emit_cycle_telemetry(cycles_per_inference_milli, have_cycles);
  emit_timer_telemetry(seconds_per_inference_scaled_1e6, have_seconds);

  /* Emit the legacy Phase 0 acceptance tokens as well so the existing smoke
   * workflow can remain useful while the richer HIL runner is introduced. */
  printf("STM32_AI_RUN=OK\r\n");
  if (have_cycles)
  {
    printf("STM32_AI_LATENCY_CYCLES=%lu\r\n", (unsigned long)cycles_per_inference_int);
  }
  else
  {
    emit_line("STM32_AI_LATENCY_CYCLES=0");
  }
  flush_stdout();

  ai_network_destroy(network);
  return 0;
}
