# What the board measures

An energy result belongs to a **candidate, board and measurement window**. Change the window and the number answers a different question, even when the model weights and board stay the same.

![Wiring diagram showing the device under test, power monitor and measurement harness](../images/hil.png)

The board under test draws power through an INA228 monitor. A separate harness reads the monitor over I²C and observes GPIO markers from the board to identify the measurement interval.

| Runtime mode | Measurement window | Question it answers |
| --- | --- | --- |
| `back_to_back` | Repeated inferences run continuously. | What does inference cost when the device stays busy? |
| `cadenced` | Scheduled windows include inference and the following sleep or wait interval. | What does a sensing period cost at this cadence? |

For a cadenced study, read the deadline-miss telemetry alongside energy. A low-energy candidate that misses its deadline may be unsuitable for deployment. Set a feasibility rule if missed deadlines should affect selection.

FLOPs and static memory traffic are available as desktop proxies. Build-reported RAM and flash usage require compilation. Energy and measured latency require the hardware path. Choose the cheapest workflow that actually supplies the metric you need.

[Choose a workflow](../run/index.md) · [Configure the target](../config/target.md)
