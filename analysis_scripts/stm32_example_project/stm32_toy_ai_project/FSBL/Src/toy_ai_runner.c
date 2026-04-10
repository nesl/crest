#include "toy_ai_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "network.h"       /* generated: AI_NETWORK_* macros, ai_network_* API */
#include "network_data.h"  /* generated: AI_NETWORK_DATA_ACTIVATIONS_SIZE, weight storage */

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

/* Zero-fill the first (and only) input tensor.
 * The input .data pointer is already set to an offset inside s_network_activations
 * by ai_network_create_and_init, so this write goes directly into the
 * activations pool.  For Phase 0 we just want a known-good input; real
 * inference would copy sensor data here instead. */
static void fill_input_buffer(ai_buffer *input)
{
  memset((void *)input[0].data, 0, AI_NETWORK_IN_1_SIZE_BYTES);
}

int toy_ai_run_once(void)
{
  ai_handle network = AI_HANDLE_NULL;

  /* Pass the activations pool to the runtime.  NULL for weights tells the
   * legacy API to use the embedded weight array from network_data_params.c
   * (the non-binary generation path — weights are compiled into the ELF,
   * so no external flash address is needed). */
  const ai_handle activations[] = {s_network_activations};
  ai_error err = ai_network_create_and_init(&network, activations, NULL);
  if (err.type != AI_ERROR_NONE)
  {
    printf("STM32_AI_INIT=FAIL type=%d code=%d\r\n", err.type, err.code);
    fflush(stdout);
    return -1;
  }

  printf("STM32_AI_INIT=OK\r\n");
  fflush(stdout);  /* newlib-nano buffers stdout; flush explicitly so the token
                    * is visible over UART before inference begins */

  /* Get pointers to the network's input and output buffer descriptors.
   * Both are backed by regions inside s_network_activations — their .data
   * fields were set during create_and_init and must not be reassigned. */
  ai_buffer *input  = ai_network_inputs_get(network, NULL);
  ai_buffer *output = ai_network_outputs_get(network, NULL);

  /* Warm-up run: drives the runtime through its full execution path once
   * before measurement.  This amortises any one-time setup cost (cache
   * warming, branch predictor state) so the timed run reflects steady-state
   * latency rather than cold-start behaviour. */
  fill_input_buffer(input);
  ai_i32 batch = ai_network_run(network, input, output);
  if (batch != 1)
  {
    printf("STM32_AI_RUN=FAIL batch=%ld\r\n", (long)batch);
    fflush(stdout);
    ai_network_destroy(network);
    return -2;
  }

  /* Timed run: DWT->CYCCNT is a free-running CPU cycle counter enabled in
   * DWT_Init() (main.c).  Capture before and after a single inference.
   * elapsed_cycles / CPU_Hz gives wall-clock latency. */
  fill_input_buffer(input);
  uint32_t start_cycles   = DWT->CYCCNT;
  batch                   = ai_network_run(network, input, output);
  uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;
  if (batch != 1)
  {
    printf("STM32_AI_RUN=FAIL batch=%ld\r\n", (long)batch);
    fflush(stdout);
    ai_network_destroy(network);
    return -3;
  }

  /* Emit the Phase 0 acceptance tokens.  The smoke test script watches for
   * all three of these lines to declare PASS. */
  printf("STM32_AI_RUN=OK\r\n");
  printf("STM32_AI_LATENCY_CYCLES=%lu\r\n", (unsigned long)elapsed_cycles);
  fflush(stdout);

  ai_network_destroy(network);
  return 0;
}
