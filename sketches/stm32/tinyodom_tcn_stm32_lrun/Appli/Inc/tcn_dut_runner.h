#ifndef TCN_DUT_RUNNER_H
#define TCN_DUT_RUNNER_H

/*
 * tcn_dut_runner — TinyODOM STM32 DUT contract
 *
 * This module turns the canonical STM32 LRUN App into a TinyODOM-compatible DUT:
 *   - waits for host START after emitting DUT READY
 *   - drives D3 active-low arm on PE9
 *   - drives D2 trigger on PD0
 *   - runs either a back-to-back or cadenced measurement phase
 *   - emits UART telemetry compatible with the repo's existing harness parser
 *
 * Key UART lines:
 *   DUT READY
 *   phase output: <back_to_back|cadenced>
 *   runs: <N>
 *   clock hz output: <cpu_clock_hz>
 *   dwt cycles per inference output: <avg_cycles>
 *   timer per inference output: <seconds_per_inference>
 *   timer per window output: <seconds_per_window>
 *
 * Legacy Phase 0 lines are preserved for continuity:
 *   STM32_AI_INIT=OK / FAIL
 *   STM32_AI_RUN=OK / FAIL
 *   STM32_AI_LATENCY_CYCLES=<avg_cycles>
 *
 * Prerequisites in main.c before calling tcn_dut_run_once():
 *   - DWT_Init()               — enables DWT->CYCCNT for cycle counting
 *   - DebugCom_Init()          — configures USART1 clock source and BSP COM1
 *   - tcn_dut_harness_gpio_init() — configures PE9/PD0 for harness signaling
 *
 * Runtime lifecycle note:
 *   - the canonical LRUN App currently runs one host-driven session and then
 *     idles in main.c until the board is reset or power-cycled
 *
 * The network used is the one staged into the generated `network*.c/.h` files
 * before each build. The generated default weight mapping may resolve to either
 * an embedded array or an external flash address depending on the selected
 * weight-storage mode.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Configure PE9/PD0 as the active-low arm / trigger outputs used by the
 * external harness. Call this once from main.c after HAL clock setup. */
void tcn_dut_harness_gpio_init(void);
void tcn_dut_rtc_irq_handler(void);

/*
 * tcn_dut_run_once — initialise, handshake, warm up, time, and report one DUT
 * attempt covering one configured measurement window.
 *
 * Returns:
 *    0   success
 *   -1   ai_network_create_and_init failed
 *   -2   warm-up ai_network_run returned unexpected batch count
 *   -3   START command timed out
 *   -4   measured ai_network_run returned unexpected batch count
 *   -5   required XSPI external-weight mapping failed
 *   -6   RTC initialization failed
 */
int tcn_dut_run_once(void);

#ifdef __cplusplus
}
#endif

#endif /* TCN_DUT_RUNNER_H */
