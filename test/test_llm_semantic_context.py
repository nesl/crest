# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Focused tests for deterministic LLM semantic context."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.model_families.audio_dscnn import AudioDSCNNFamily  # noqa: E402
from crest.model_families.odom_tcn import OdomTCNFamily  # noqa: E402
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request  # noqa: E402
from crest.optimizers.llm.search_space import (  # noqa: E402
    SearchParam,
    SearchSpaceDescriptor,
)
from crest.optimizers.llm.semantic_context import (  # noqa: E402
    SEMANTIC_CONTEXT_VERSION,
    build_semantic_context,
)
from crest.pipeline_types import (  # noqa: E402
    DataSplit,
    DatasetBundle,
    ModelBuildContext,
    TargetSpec,
    TaskMetricContract,
)


def _bundle(
    *,
    input_shape: tuple[int, ...],
    input_dtype: str,
    metadata: dict | None = None,
) -> DatasetBundle:
    """Build a normalized dataset bundle without retaining example data."""
    return DatasetBundle(
        train=DataSplit(inputs=None, targets=None),
        input_shape=input_shape,
        input_dtype=input_dtype,
        metadata=dict(metadata or {}),
    )


def _semantic_record(
    *,
    dataset_name: str = "oxiod",
    dataset_config: Dict | None = None,
    bundle: DatasetBundle | None = None,
    target_spec: TargetSpec | None = None,
    metric_contract: TaskMetricContract | None = None,
    descriptor: SearchSpaceDescriptor | None = None,
    model_family_name: str = "odom_tcn",
    task_name: str = "odometry_regression",
    device_config: Dict | None = None,
    training_config: Dict | None = None,
    enabled: bool = True,
) -> dict:
    """Build representative context from concrete CREST runtime dataclasses."""
    bundle = bundle or _bundle(
        input_shape=(200, 6),
        input_dtype="float32",
        metadata={
            "sampling_rate_hz": 100,
            "window_size": 200,
            "stride": 20,
            "input_dim": 6,
        },
    )
    target_spec = target_spec or TargetSpec(
        task_type="regression",
        output_names=["velx", "vely"],
        output_shapes=[(1,), (1,)],
    )
    metric_contract = metric_contract or TaskMetricContract(
        available_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        training_only_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        nonnegative_metric_names={"rmse_vel_x", "rmse_vel_y", "rmse_total"},
        primary_metric_names={"rmse_total"},
    )
    build_context = ModelBuildContext(
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        target_spec=target_spec,
        dataset_metadata=bundle.metadata,
    )
    descriptor = descriptor or SearchSpaceDescriptor(
        tuple(OdomTCNFamily().trial_search_space(build_context, Dict()))
        + (
            SearchParam(
                "quantization_mode",
                "categorical",
                choices=("float", "int8_ptq"),
                description="Controls deployment numeric representation.",
            ),
            SearchParam(
                "cpu_clock_mhz_index",
                "int",
                low=0,
                high=1,
                description="Selects a configured MCU clock.",
            ),
        )
    )
    return build_semantic_context(
        enabled=enabled,
        dataset_name=dataset_name,
        dataset_config=dataset_config
        or Dict(directory="/secret/dataset", sampling_rate_hz=100, window_size=200, stride=20),
        dataset_bundle=bundle,
        model_build_context=build_context,
        task_name=task_name,
        target_spec=target_spec,
        metric_contract=metric_contract,
        score_config=Dict(
            type="multi-objective",
            params=Dict(
                objectives=[
                    Dict(metric="rmse_total", direction="minimize"),
                    Dict(metric="energy_mj_per_inference", direction="minimize"),
                ]
            ),
        ),
        feasibility_config=Dict(
            train_if_infeasible=False,
            rules=[
                Dict(
                    rule="ram_limit",
                    metric="ram_bytes",
                    condition="gt",
                    reference=Dict(type="metric", metric="max_ram_bytes"),
                    reason="RAM must fit",
                )
            ],
        ),
        model_family_name=model_family_name,
        descriptor=descriptor,
        device_config=device_config
        or Dict(
            name="STM32_NUCLEO_N657X0_Q",
            hil=False,
            compile_when_hil_disabled="false",
            serial_port="/dev/secret",
            cpu_clock_mhz_options=[64, 128],
            stm32=Dict(weight_storage_mode="external_flash"),
            api_key="do-not-log",
        ),
        training_config=training_config
        or Dict(
            train=False,
            quantization=Dict(
                mode="float",
                search=True,
                choices=["float", "int8_ptq"],
            ),
        ),
        collect_compile_metrics=False,
    )


class SemanticContextTests(unittest.TestCase):
    """Validate bounded derivation, ablation, and provenance."""

    def test_odom_context_contains_imu_window_task_and_device_meaning(self) -> None:
        """Odom context is derived from normalized runtime contracts."""
        record = _semantic_record()
        payload = record["payload"]

        self.assertEqual(record["version"], SEMANTIC_CONTEXT_VERSION)
        self.assertEqual(payload["dataset_context"]["modality"], "inertial_measurement_unit")
        self.assertEqual(payload["dataset_context"]["sampling_rate_hz"], 100)
        self.assertEqual(payload["dataset_context"]["window_size"], 200)
        self.assertEqual(payload["dataset_context"]["stride"], 20)
        self.assertEqual(payload["dataset_context"]["input_shape"], [200, 6])
        self.assertEqual(payload["task_context"]["type"], "regression")
        self.assertEqual(
            [item["name"] for item in payload["task_context"]["outputs"]],
            ["velx", "vely"],
        )
        self.assertEqual(
            payload["task_context"]["objective"]["study_outputs"][0],
            {"metric": "rmse_total", "direction": "minimize"},
        )
        self.assertGreater(payload["device_context"]["ram_capacity_bytes"], 0)
        self.assertGreater(payload["device_context"]["flash_capacity_bytes"], 0)
        self.assertEqual(payload["device_context"]["cpu_clock_mhz_choices"], [64, 128])
        self.assertEqual(
            payload["device_context"]["quantization_choices"],
            ["float", "int8_ptq"],
        )
        self.assertEqual(
            payload["device_context"]["weight_storage_mode"],
            "external_flash",
        )

    def test_audio_context_contains_classification_and_input_features(self) -> None:
        """Audio semantics expose log-mel shape and class-output meaning."""
        bundle = _bundle(
            input_shape=(201, 64),
            input_dtype="float32",
            metadata={
                "feature_kind": "log_mel_power",
                "sample_rate_hz": 16000,
                "mel_bins": 64,
                "expected_frames": 201,
                "window_ms": 25,
                "hop_ms": 10,
            },
        )
        target = TargetSpec(
            task_type="classification",
            output_names=["class_logits"],
            output_shapes=[(10,)],
            metadata={
                "num_classes": 10,
                "label_encoding": "class_index",
                "from_logits": True,
            },
        )
        build_context = ModelBuildContext(
            input_shape=bundle.input_shape,
            input_dtype=bundle.input_dtype,
            target_spec=target,
            dataset_metadata=bundle.metadata,
        )
        descriptor = SearchSpaceDescriptor(
            AudioDSCNNFamily().trial_search_space(
                build_context,
                Dict(family="audio_dscnn", params=Dict(), search=Dict()),
            )
        )
        record = _semantic_record(
            dataset_name="urbansound8k_mel",
            bundle=bundle,
            target_spec=target,
            metric_contract=TaskMetricContract(
                available_metric_names={"loss", "accuracy", "macro_f1"},
                primary_metric_names={"accuracy", "macro_f1"},
            ),
            descriptor=descriptor,
            model_family_name="audio_dscnn",
            task_name="sound_classification",
            device_config=Dict(name="ARDUINO_NANO_33_BLE_SENSE", hil=False),
            training_config=Dict(train=False, quantization=Dict(mode="float", search=False)),
        )
        payload = record["payload"]

        self.assertEqual(payload["dataset_context"]["modality"], "log_mel_audio")
        self.assertEqual(payload["dataset_context"]["feature_kind"], "log_mel_power")
        self.assertEqual(payload["dataset_context"]["input_shape"], [201, 64])
        self.assertEqual(payload["dataset_context"]["mel_bins"], 64)
        self.assertEqual(payload["task_context"]["type"], "classification")
        self.assertEqual(payload["task_context"]["num_classes"], 10)
        self.assertEqual(payload["task_context"]["outputs"][0]["name"], "class_logits")
        self.assertEqual(
            set(payload["parameter_semantics"]),
            set(descriptor),
        )

    def test_missing_optional_metadata_is_omitted_and_sensitive_values_never_leak(self) -> None:
        """The whitelist excludes paths, ports, secrets, and absent fields."""
        bundle = _bundle(input_shape=(8, 3), input_dtype="float32")
        target = TargetSpec(task_type="regression", output_names=[], output_shapes=[])
        descriptor = SearchSpaceDescriptor((SearchParam("legacy", "int", low=1, high=2),))
        record = _semantic_record(
            dataset_name="third_party",
            dataset_config=Dict(directory="sensitive/data", cache_dir="sensitive/cache"),
            bundle=bundle,
            target_spec=target,
            metric_contract=TaskMetricContract(),
            descriptor=descriptor,
            model_family_name="third_party",
            task_name="third_party_task",
            device_config=Dict(
                name="UNKNOWN_BOARD",
                serial_port="/dev/ttySECRET",
                api_key_env="SECRET_ENV",
                api_key="actual-secret",
                hil=False,
            ),
            training_config=Dict(train=False, quantization=Dict()),
        )
        encoded = json.dumps(record, sort_keys=True)

        self.assertNotIn("modality", record["payload"]["dataset_context"])
        self.assertNotIn("ram_capacity_bytes", record["payload"]["device_context"])
        self.assertEqual(record["payload"]["parameter_semantics"], {})
        for forbidden in (
            "sensitive/data",
            "sensitive/cache",
            "/dev/ttySECRET",
            "SECRET_ENV",
            "actual-secret",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_output_and_hash_are_deterministic_and_schema_owns_legal_values(self) -> None:
        """Canonical payload hashing is stable and semantics do not copy bounds."""
        first = _semantic_record()
        second = _semantic_record()

        self.assertEqual(first, second)
        self.assertEqual(len(first["hash"]), 64)
        for semantics in first["payload"]["parameter_semantics"].values():
            self.assertTrue({"low", "high", "choices", "kind"}.isdisjoint(semantics))

        descriptor = SearchSpaceDescriptor((SearchParam("width", "int", low=3, high=9),))
        disabled = _semantic_record(descriptor=descriptor, enabled=False)
        context = PromptContext(
            study_name="ablation",
            model_family="third_party",
            descriptor=descriptor,
            objective_summary="maximize score",
            attempted_trials=4,
            feasible_completed_trials=2,
            target_feasible_trials=8,
            max_total_attempts=12,
            batch_size=1,
            recent_trials=({"number": 3, "params": {"width": 5}},),
            semantic_context=disabled,
        )
        request = build_candidate_request(context, prompt_version="v1")
        prompt = json.loads(request.user_prompt)

        self.assertEqual(prompt["search_space"]["width"], {"kind": "int", "low": 3, "high": 9})
        self.assertEqual(prompt["recent_trials"][0]["params"], {"width": 5})
        self.assertEqual(prompt["trial_budget"]["attempted"], 4)
        self.assertFalse(prompt["semantic_context"]["enabled"])
        for omitted in (
            "dataset_context",
            "task_context",
            "model_context",
            "device_context",
            "runtime_context",
        ):
            self.assertNotIn(omitted, prompt)
        self.assertFalse(request.metadata["semantic_context_enabled"])
        self.assertEqual(request.metadata["semantic_context_version"], "v1")
        self.assertEqual(request.metadata["semantic_context_hash"], disabled["hash"])

    def test_enabled_prompt_places_semantics_on_existing_search_schema(self) -> None:
        """Descriptions augment, rather than replace, the legal descriptor schema."""
        descriptor = SearchSpaceDescriptor(
            (
                SearchParam(
                    "width",
                    "int",
                    low=2,
                    high=8,
                    description="Controls model width.",
                    units="channels",
                    typical_effects=("Width typically affects capacity and compute.",),
                ),
            )
        )
        record = _semantic_record(descriptor=descriptor)
        context = PromptContext(
            study_name="semantics",
            model_family="example",
            descriptor=descriptor,
            objective_summary="maximize score",
            attempted_trials=0,
            feasible_completed_trials=0,
            target_feasible_trials=1,
            max_total_attempts=2,
            batch_size=1,
            semantic_context=record,
        )
        prompt = json.loads(build_candidate_request(context, prompt_version="v1").user_prompt)

        self.assertEqual(prompt["search_space"]["width"]["low"], 2)
        self.assertEqual(prompt["search_space"]["width"]["high"], 8)
        self.assertEqual(prompt["search_space"]["width"]["units"], "channels")
        self.assertIn("priors", prompt["semantic_context"]["guidance"][0])
        self.assertIn("observations take precedence", prompt["semantic_context"]["guidance"][1])
        self.assertIn("exact raw Optuna", prompt["semantic_context"]["guidance"][2])


if __name__ == "__main__":
    unittest.main()
