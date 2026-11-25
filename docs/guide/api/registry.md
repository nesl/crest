# Registration and selection

Each software registry maps a configured name to a component class. The [extension guides](../extend/index.md) show where to register components and how to select them in a run.

::: crest.registry.ComponentRegistry
    options:
      show_root_heading: true
      members: [register, get, names]

::: crest.component_selection.resolve_component_selection
    options:
      show_root_heading: true

::: crest.builtin_components.ensure_builtin_components_registered
    options:
      show_root_heading: true

::: crest.runtime_bootstrap.bootstrap_pipeline
    options:
      show_root_heading: true

::: crest.runtime_bootstrap.BootstrappedPipeline
    options:
      show_root_heading: true
