# CREST ESP32-S3 TFLite Micro runner

This is the inference-only ESP-IDF project used by the
`FREENOVE_ESP32_S3_WROOM` backend. CREST generates `model_data.cc`,
`model_data.h`, `fixtures.h`, `crest_manifest.h`, and a raw
`crest_candidate.tflite` under `main/`; these files and the IDF build cache are
ignored. The committed runner allocates one tensor arena in PSRAM, registers
the operators used by current FER exports, recopies every prepared input before
invoke, and emits classifier-only timing plus identity evidence.

The fixed profile derives from the saved 2026-10-30 deployment: ESP-IDF 5.5.1,
ESP-TFLM 1.3.3~1, CPU 240 MHz, 8 MiB DIO/40 MHz flash, 8 MiB octal/40 MHz
PSRAM, debug optimization with ESP-NN, and the 4 MiB factory partition in
`partitions-4m.csv`. CREST uses `app-flash`; it does not rewrite the bootloader
or partition table during candidate evaluation.

GPIO 21 drives the harness D2 trigger high during the activity interval. GPIO
47 drives harness D3 arm low before the trigger and high afterward. Harness
firmware defines D2 as an input pulldown and D3 as an active-low input pullup.
The activity interval includes control work and must not be used as classifier
latency; `Invoke()` samples in serial telemetry are authoritative.

The Freenove documentation reviewed for the target was pinned at upstream
revision `3563960643987e327a9213c90566fca1e8c5ab67`: `Start Here.pdf`, relevant
programming and pin-restriction sections of `C/C_Tutorial.pdf`,
`ESP32S3_Pinout.png`, and `LICENSE.txt`. The supplied tree contains no schematic
source. Saved esptool and IDF records identify the deployed black N8R8 variant;
those records and a hardware-free compile do not constitute current-board flash
or electrical validation. No Freenove source code is copied into this project.

## Hardware-free reference compile, 2026-11-11

The untrained reference96 component artifact was staged with
`stage_fixture_header` and `convert_to_cpp_model`, then compiled without upload
or board access. The input hashes were:

- TFLite: `365a743e14d802fad4911f7c132cf3431871ec22c6b4bd2b78060f36d0dbcade`
- representative96 NPZ: `993b81587a043fd3f23b02a3a24799455856ad413dac80a5b04eab75518e022c`

After activating the saved ESP-IDF 5.5.1 environment, the backend's build step
is reproducible as:

```sh
idf.py -C firmware/esp32/crest_esp32_tflm \
  -B firmware/esp32/crest_esp32_tflm/build \
  -DCREST_AGGREGATE_MEDIAN=1 \
  -DCREST_DUT_ARM_HOLD_MS=600 \
  -DCREST_ESP_NN_ENABLED=1 \
  -DCREST_HARNESS_ARM_PIN=47 \
  -DCREST_HARNESS_STABLE_LOW_MS=500 \
  -DCREST_HARNESS_TRIGGER_PIN=21 \
  -DCREST_INFERENCE_RUNS=5 \
  -DCREST_OPTIMIZE_FOR_SIZE=0 \
  -DCREST_TENSOR_ARENA_BYTES=2097152 \
  -DCREST_TENSOR_ARENA_IN_PSRAM=1 \
  -DCREST_WARMUP_RUNS=1 \
  build size
```

The resolved target was `esp32s3`; the image was 629,937 bytes, static DIRAM
was 63,059 bytes, and static IRAM was 16,384 bytes. The retained application
was 630,048 bytes with SHA-256
`23f290bd88f804dcfe078e74f8a16f8df3e8146b1dd516d450e4a38623be1ac1` and
contained the exact TFLite bytes once. The resolved sdkconfig SHA-256 was
`d065c95d6f38504fbb3f49d25793923097902752f9b3acedb8aee82428935150`.
This is a compile and artifact-identity result; it is not an on-board timing or
correctness result.

## Physical acceptance, 2026-11-11

The firmware content committed at `1275021acfa858751397530ecac9c0b1f49264bb`
was tested on the attached ESP32-S3 revision 0.2 using application-only flashes.
Read-only inspection reported an 8 MiB flash, 8 MiB embedded PSRAM, 40 MHz
crystal, disabled secure boot and flash encryption, and the expected 4 MiB
factory application partition at `0x10000`.

The untrained reference96 artifact and the ten ordered representative96 inputs
used the hashes recorded above. With one warmup and two measured invokes per
input, the first 2 MiB-arena run produced 20 ordered samples, ten output
vectors, valid identity evidence, 363,212 arena bytes used, and a median invoke
latency of 1.5600605 seconds. All ten raw int8 output vectors exactly matched
host TFLite execution.

A separate 64 KiB-arena run failed as expected: TFLite Micro requested 345,488
bytes with 47,952 available and reported 297,536 missing. The failure-state
handshake re-emitted the allocation error after `START`, and the controller
classified the singleton attempt as arena exhausted. A final separate 2 MiB
run restored the board to good firmware and again produced 20 valid samples,
a 1.560188 second median, and exact raw int8 output parity for all ten inputs.
The harness GPIOs carried activity signaling during these runs; no harness
energy result or campaign timing claim was produced.
