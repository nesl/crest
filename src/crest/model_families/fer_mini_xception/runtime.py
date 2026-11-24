"""Select cached FER pipelines through the existing bootstrap contract."""
from copy import deepcopy
from crest.component_selection import cfg_get
from crest.runtime_bootstrap import bootstrap_pipeline


def resolution_pipeline(owner, hparams):
    """Return a cached pipeline for an explicitly enabled FER resolution."""
    config = owner.config
    if cfg_get(cfg_get(config, "model", {}), "family", None) != "fer_mini_xception":
        return None
    choices = cfg_get(config.model.search, "resolution", None)
    if choices is None:
        if "resolution" in hparams:
            raise ValueError("Resolution search is not enabled in this configuration.")
        return None
    value = hparams.get("resolution")
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise ValueError("Candidate resolution is not allowed by model.search.resolution.")
    if cfg_get(config.dataset, "name", None) != "ferplus_rafdb_mixed":
        raise ValueError("FER resolution search requires the mixed FER dataset.")
    cache = getattr(owner, "_fer_resolution_pipelines", None)
    if cache is None:
        cache = owner._fer_resolution_pipelines = {}
    if value not in cache:
        selected = deepcopy(config)
        selected.dataset.params.image_size = value
        bundle = getattr(owner, "dataset_bundle", None)
        if bundle is not None and tuple(bundle.input_shape or ()) == (value, value):
            cache[value] = bootstrap_pipeline(selected, dataset=owner.dataset, bundle=bundle)
        else:
            cache[value] = bootstrap_pipeline(selected)
    return cache[value]


def attach_pipeline(owner, pipeline):
    """Attach the existing bootstrap result on either client or server."""
    for name in ("dataset", "task", "model_family", "target_spec", "metric_contract", "model_build_context"):
        setattr(owner, name, getattr(pipeline, name))
    owner.dataset_bundle = pipeline.bundle
    for name in ("dataset_name", "task_name", "model_family_name", "dataset_config", "task_config", "model_config"):
        setattr(owner, name, pipeline.selection[name])
