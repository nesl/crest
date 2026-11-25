# Add a board

A board backend stages an exported model, compiles it, uploads it, and returns runtime measurements through the [device interface](../api/devices.md).

| Toolchain | Starting point |
| --- | --- |
| Arduino CLI | Use `src/crest/microcontrollers/arduino_ble33.py` for a fixed FQBN or `arduino_portenta_h7.py` for board options. Both wrap `ArduinoDevice`, which already stages the sketch and handles compile, upload, and measurement. |
| STM32 Cube and ST Edge AI | Start from `stm32_nucleo_n657x0.py`. Check project staging, generation, ST-LINK loading, and the serial protocol against the new target. The current backend is built around the Nucleo N657X0-Q project and runtime contract. |
| Other toolchains | Implement `DeviceInterface` in a dedicated backend and provide the build, upload, and runtime integration. |

1. Supply a `DeviceSpec` with the board identifier, RAM and flash limits, arena candidates, and any required toolchain metadata. Make capability flags match the workflow your backend supports.
2. Implement or inherit `spec`, `compile`, `upload`, `measure`, and `evaluate`. Add candidate staging and cleanup if the backend needs a generated project. Return structured result objects for the shared controller.
3. Add an entry in `_registry_entries()` in `src/crest/microcontrollers/__init__.py`. Each entry maps a board name to a module path, wrapper class name, and default-spec symbol. The factory passes `serial_port` and `device_options` as keyword arguments. If you add board-specific settings, normalize them in `resolve_device_options`.
4. Select the registry key in `device.name`. For the shipped Nano 33 BLE Sense wrapper, the key is `ARDUINO_NANO_33_BLE_SENSE`.

Use mocked tool and serial responses to check timeout handling, telemetry parsing, and failure results. Then compile a candidate with the actual toolchain, upload it to a connected board, and verify runtime metrics. Measure energy with the CREST harness and INA228 telemetry path. See `test/test_stm32_backend.py` and `test/test_stm32_runtime.py` for examples.
