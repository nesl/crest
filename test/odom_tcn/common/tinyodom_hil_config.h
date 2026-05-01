#pragma once

#ifndef TINYODOM_INFERENCE_RUNS
#define TINYODOM_INFERENCE_RUNS 10
#endif

// GPIO pin shared between DUT and harness for start/stop signaling.
#ifndef TINYODOM_HARNESS_TRIGGER_PIN
#define TINYODOM_HARNESS_TRIGGER_PIN 2
#endif

// GPIO pin used by DUT to arm/disarm the harness (active-low).
#ifndef TINYODOM_HARNESS_ARM_PIN
#define TINYODOM_HARNESS_ARM_PIN 3
#endif

// DUT holds arm LOW for this duration before raising the trigger pin.
#ifndef TINYODOM_DUT_ARM_HOLD_MS
#define TINYODOM_DUT_ARM_HOLD_MS 600
#endif

// How long the harness must see a stable LOW before accepting a rising edge.
#ifndef TINYODOM_HARNESS_STABLE_LOW_MS
#define TINYODOM_HARNESS_STABLE_LOW_MS 500
#endif

// Harness-side ARM timeout (waiting for a valid rising edge). Use 0 to disable.
#ifndef TINYODOM_HARNESS_ARM_TIMEOUT_MS
#define TINYODOM_HARNESS_ARM_TIMEOUT_MS 600000
#endif

// Harness-side ACTIVE timeout (waiting for the falling edge). Use 0 to disable.
#ifndef TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS
#define TINYODOM_HARNESS_ACTIVE_TIMEOUT_MS 600000
#endif

// Protocol strings shared by DUT, harness, and host.
#define TINYODOM_DUT_READY "DUT READY"
#define TINYODOM_HARNESS_READY "HARNESS READY"
#define TINYODOM_CMD_PING "PING"
#define TINYODOM_CMD_START "START"
#define TINYODOM_RESP_ARMED "ARMED"
#define TINYODOM_RESP_DONE "DONE"
