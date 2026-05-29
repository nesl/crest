# Case Study Configs

These configs are the paper-oriented run setups for the CREST case studies.
Use the top-level configs in `src/config/` as general examples; use this folder
when you want the specific study axes from the paper.

- **Case Study 1:** OxIOD/TCN measured-energy NAS across BLE33, Portenta M7,
  Portenta M4, and STM32. The FLOPs and memory-proxy examples live one
  directory up as `nas_config_flops_rmse.yaml` and
  `nas_config_memory_proxy.yaml`.
- **Case Study 2:** STM32/OxIOD schedule comparison. The back-to-back config
  is the continuous-inference side; the cadenced config is the sensing-window
  side.
- **Case Study 3:** UrbanSound8K/DS-CNN application-level scoring on Portenta
  M7 and STM32.

Each YAML includes its intended launch command near the top.
