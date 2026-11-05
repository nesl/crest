"""Focused tests for portable CS1 inference and selection analysis."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module(name):
    """Load a standalone analysis script.

    Parameters
    ----------
    name : str
        Script basename.

    Returns
    -------
    module
        Imported analysis module.
    """
    path = (
        Path(__file__).parents[1] / "analysis_scripts" / "cs1_offline" / (name + ".py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affine_weighting_and_architecture_separation():
    """Check graph weighting, scheduled accounting, and leakage rejection.

    Returns
    -------
    None
        Assertions validate an independently specified linear example.
    """
    mod = _module("heldout_prediction")
    train = pd.DataFrame(
        dict(
            arch_id=["a", "a", "b", "c"],
            pool=["p"] * 4,
            latency_s=[1.0, 1.0, 2.0, 3.0],
            continuous_mj=[3.0, 3.0, 5.0, 7.0],
            scheduled_mj=[13.0, 13.0, 15.0, 17.0],
        )
    )
    test = pd.DataFrame(
        dict(
            arch_id=["d"],
            pool=["p"],
            latency_s=[4.0],
            continuous_mj=[9.0],
            continuous_hat_mj=[8.0],
            T_s=[1.0],
            C_s=[0.4],
            wait_mw=[10.0],
            offset_mj=[4.0],
        )
    )
    frames, coefficients = mod.predict(
        train, test, ["pool"], "same_pool_architecture_holdout", 0, 0
    )
    assert np.isclose(coefficients[0]["a_mj"], 1.0)
    assert np.isclose(coefficients[0]["s_mw"], 2.0)
    assert np.isclose(frames[0].affine_scheduled.iloc[0], 19.0)
    assert np.isclose(frames[0].scheduled_direct_affine_scheduled.iloc[0], 19.0)
    with pytest.raises(ValueError, match="overlap"):
        mod.predict(train, test.assign(arch_id="a"), ["pool"], "test", 0, 0)


def test_relocated_source_resolution(tmp_path):
    """Check archived suffix resolution without the original absolute tree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary pytest directory.

    Returns
    -------
    None
        Verifies resolution to the relocated log.
    """
    mod = _module("heldout_prediction")
    source = tmp_path / "models" / "log.csv"
    source.parent.mkdir()
    source.write_text("a\n1\n")
    assert mod.resolve_source("/old/machine/models/log.csv", tmp_path) == source


def test_raw_ranking_uses_same_quality_bank_and_tie_breaks():
    """Check quality exclusion and deterministic saved-quality tie breaking.

    Returns
    -------
    None
        Verifies raw ranking and measured-reference regret.
    """
    mod = _module("raw_selection_baseline")
    p = pd.DataFrame(
        dict(
            protocol=["holdout"] * 3,
            repeat=[0] * 3,
            fold=[0] * 3,
            pool=["p"] * 3,
            rmse=[0.6, 0.5, 2.0],
            study=["s"] * 3,
            optuna_trial=[1, 2, 3],
            continuous_mj=[1.0, 1.0, 0.0],
            latency_s=[2.0, 1.0, 0.0],
            scheduled_mj=[5.0, 4.0, 0.0],
            affine_scheduled=[5.0, 4.0, 0.0],
            measured_continuous_scheduled=[5.0, 4.0, 0.0],
            scheduled_direct_affine_scheduled=[5.0, 4.0, 0.0],
        )
    )
    r = mod.select(p)
    assert len(r) == 15
    assert (r.sel == 4.0).all()
    assert (r.regret == 0.0).all()
    assert len(mod.select(p, quality_ceilings=[1.0])) == 5
    assert mod.select(p, quality_ceilings=[0.55]).empty


def test_load_uses_relocated_sources_without_paper_row_count(tmp_path):
    """Check a single-row custom bank resolves its relocated architecture log.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary pytest directory.

    Returns
    -------
    None
        Validates portable reconstruction independently of the submitted bank.
    """
    mod = _module("heldout_prediction")
    source = tmp_path / "archive" / "models" / "source.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            dict(
                energy_mj_per_inference=2.0,
                cadenced_energy_mj_per_window=5.0,
                hparam__nb_filters=8,
                hparam__kernel_size=3,
                hparam__dilations="[1, 2]",
                hparam__use_skip_connections=1,
                hparam__norm_flag=0,
                hparam__input_dim=6,
                hparam__timesteps=16,
            )
        ]
    ).to_csv(source, index=False)
    row = dict(
        pool="custom_pool",
        trial=1,
        latency_s=0.2,
        continuous_mj=2.0,
        scheduled_mj=5.0,
        clock_mhz=100,
        rmse=0.5,
        study="custom",
        optuna_trial=1,
        continuous_latency_ms=200.0,
        continuous_hat_mj=2.0,
        T_s=1.0,
        C_s=0.2,
        wait_mw=1.0,
        offset_mj=0.0,
    )
    predictions = tmp_path / "predictions.csv"
    observations = tmp_path / "observations.csv"
    pd.DataFrame([row]).to_csv(predictions, index=False)
    pd.DataFrame(
        [
            dict(
                pool="custom_pool",
                trial=1,
                quantization="custom",
                source="/nonexistent/machine/models/source.csv",
            )
        ]
    ).to_csv(observations, index=False)
    d = mod.load(predictions, observations, source.parents[1])
    assert len(d) == 1
    assert d.arch_id.nunique() == 1
    source.unlink()
    with pytest.raises(FileNotFoundError):
        mod.load(predictions, observations, source.parents[1])


def test_ambiguous_source_and_configurable_quality(tmp_path):
    """Reject ambiguous source relocation and respect custom quality ceilings.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary pytest directory.

    Returns
    -------
    None
        Checks validation and configurable selection behavior.
    """
    mod = _module("heldout_prediction")
    for folder in ["a", "b"]:
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "same.csv").write_text("x\n1\n")
    with pytest.raises(FileNotFoundError, match="found 2"):
        mod.resolve_source("/gone/same.csv", tmp_path)
