# Device API

See [Add a board](../extend/boards.md) for implementation and registration steps.

::: crest.microcontrollers.get_device
    options:
      show_root_heading: true

::: crest.microcontrollers.resolve_device_options
    options:
      show_root_heading: true

::: crest.devices.DeviceInterface
    options:
      show_root_heading: true
      members: [spec, prepare_candidate, cleanup_prepared_candidate, compile, upload, measure, evaluate]

::: crest.devices.DeviceSpec
    options:
      show_root_heading: true

::: crest.devices.CandidatePrepareRequest
    options:
      show_root_heading: true

::: crest.devices.CompileResult
    options:
      show_root_heading: true
      members: false

::: crest.devices.UploadResult
    options:
      show_root_heading: true

::: crest.devices.MeasureResult
    options:
      show_root_heading: true

::: crest.devices.DeviceMetrics
    options:
      show_root_heading: true
