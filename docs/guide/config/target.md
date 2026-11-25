# Target and schedule

Choose the measurements you need before choosing the hardware settings. A desktop study can still request compilation if the scoring policy references compiler-reported metrics.

| Goal | `device.hil` | `device.compile_when_hil_disabled` |
| --- | --- | --- |
| Pure desktop search with task metrics and FLOPs | `false` | `false` |
| Fetch compiler-reported RAM or flash for selected metrics | `false` | `auto` |
| Collect measured latency or board energy | `true` | Applies only when HIL is disabled |

For measured runs, set `device.name` to the target backend and `device.runtime_mode` to `back_to_back` or `cadenced`. In cadenced mode CREST can derive the latency budget from dataset timing, or you can give a positive `device.latency_budget_ms`.

??? info "Device, harness and backend-specific fields"
    <!-- include: src/config/README.md#`device` -->

[Understand measurement windows](../concepts/measurement.md) · [Bring up hardware](../run/hardware.md)
