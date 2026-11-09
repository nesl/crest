<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix: measurement context

The [micro-workload probe](experiments/micro_workload_energy_probe/README.md)
measures fixed-duration sleep, wait, polling, floating-point, and integer-compute
windows on supported boards. Use it to collect the measurement-context
appendix’s synthetic workload data.

The guide covers board setup, commands, and per-attempt and aggregate outputs.
These measurements are separate from the model-specific
[CS1 waiting-power experiments](../cs1/experiments/waiting_power/README.md).
Run one experiment at a time on a DUT/harness.
