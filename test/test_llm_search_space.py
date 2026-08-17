# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the declarative raw Optuna search-space descriptor."""

import sys
import unittest
from pathlib import Path

from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.model_families.audio_dscnn import (  # noqa: E402
    AUDIO_DSCNN_SEARCH_CHOICES,
    AUDIO_DSCNN_SEARCH_SEMANTICS,
    AudioDSCNNFamily,
)
from crest.model_families.odom_tcn import (  # noqa: E402
    DILATION_CANDIDATES,
    DROP_RATE_CHOICES,
    OdomTCNFamily,
)
from crest.optimizers.llm.search_space import (  # noqa: E402
    SearchParam,
    build_search_space_descriptor,
)
from crest.pipeline_types import ModelBuildContext, TargetSpec  # noqa: E402


def odom_context() -> ModelBuildContext:
    """Build the Odom TCN model context used by descriptor tests."""
    return ModelBuildContext(
        input_shape=(20, 6),
        input_dtype="float32",
        target_spec=TargetSpec(
            task_type="regression",
            output_names=["velx", "vely"],
            output_shapes=[(1,), (1,)],
        ),
    )


def audio_context() -> ModelBuildContext:
    """Build the audio classification context used by descriptor tests."""
    return ModelBuildContext(
        input_shape=(201, 64),
        input_dtype="float32",
        target_spec=TargetSpec(
            task_type="classification",
            output_names=["class_logits"],
            output_shapes=[(10,)],
            metadata={"num_classes": 10, "label_encoding": "class_index", "from_logits": True},
        ),
    )


def runner_config(
    *,
    train: bool = False,
    quantization_search: bool = False,
    cpu_clock_mhz_options=None,
) -> Dict:
    """Build the normalized runner config fields consumed by the descriptor."""
    return Dict(
        training=Dict(
            train=train,
            quantization=Dict(
                mode="int8_ptq",
                search=quantization_search,
                choices=["float", "int8_ptq"],
            ),
        ),
        device=Dict(cpu_clock_mhz_options=cpu_clock_mhz_options),
    )


class SearchSpaceDescriptorTests(unittest.TestCase):
    """Prove descriptors match the current family and runner sampling surface."""

    def test_odom_descriptor_uses_raw_optuna_keys_and_current_ranges(self) -> None:
        """Odom exposes the index field and exact legacy bounds and choices."""
        descriptor = build_search_space_descriptor(
            OdomTCNFamily(),
            odom_context(),
            {},
            runner_config(),
            collect_compile_metrics=False,
        )

        self.assertEqual(
            set(descriptor),
            {
                "nb_filters",
                "kernel_size",
                "dropout_rate",
                "use_skip_connections",
                "norm_flag",
                "dilations_index",
            },
        )
        self.assertNotIn("dilations", descriptor)
        self.assertEqual((descriptor["nb_filters"].low, descriptor["nb_filters"].high), (2, 63))
        self.assertEqual((descriptor["kernel_size"].low, descriptor["kernel_size"].high), (2, 15))
        self.assertEqual(
            (descriptor["dilations_index"].low, descriptor["dilations_index"].high),
            (0, len(DILATION_CANDIDATES) - 1),
        )
        self.assertEqual(descriptor["dropout_rate"].choices, tuple(DROP_RATE_CHOICES))
        self.assertEqual(descriptor["use_skip_connections"].choices, (True, False))
        self.assertEqual(descriptor["norm_flag"].choices, (True, False))

    def test_audio_descriptor_preserves_defaults_and_ordered_overrides(self) -> None:
        """Audio exposes every categorical and honors normalized config choices."""
        family = AudioDSCNNFamily()
        config = Dict(
            family="audio_dscnn",
            params=Dict(),
            search=Dict(base_channels=[24, 8], channel_growth=[2, 1.5]),
        )
        descriptor = build_search_space_descriptor(
            family,
            audio_context(),
            config,
            runner_config(),
            collect_compile_metrics=False,
        )

        self.assertEqual(tuple(descriptor), tuple(AUDIO_DSCNN_SEARCH_CHOICES))
        self.assertTrue(all(param.kind == "categorical" for param in descriptor.params))
        self.assertEqual(descriptor["base_channels"].choices, (24, 8))
        self.assertEqual(descriptor["channel_growth"].choices, (2.0, 1.5))
        self.assertEqual(
            descriptor["stride_schedule"].choices,
            AUDIO_DSCNN_SEARCH_CHOICES["stride_schedule"],
        )

    def test_runner_params_are_present_only_when_current_sampling_paths_are_active(self) -> None:
        """Quantization and CPU-clock fields follow objective activation rules."""
        family = OdomTCNFamily()
        context = odom_context()

        inactive = build_search_space_descriptor(
            family,
            context,
            {},
            runner_config(
                train=False,
                quantization_search=True,
                cpu_clock_mhz_options=[250, 400],
            ),
            collect_compile_metrics=False,
        )
        self.assertNotIn("quantization_mode", inactive)
        self.assertNotIn("cpu_clock_mhz_index", inactive)

        training_active = build_search_space_descriptor(
            family,
            context,
            {},
            runner_config(train=True, quantization_search=True),
            collect_compile_metrics=False,
        )
        self.assertEqual(
            training_active["quantization_mode"].choices,
            ("float", "int8_ptq"),
        )
        self.assertNotIn("cpu_clock_mhz_index", training_active)

        compile_active = build_search_space_descriptor(
            family,
            context,
            {},
            runner_config(
                train=False,
                quantization_search=True,
                cpu_clock_mhz_options=[250, 400, 600],
            ),
            collect_compile_metrics=True,
        )
        self.assertIn("quantization_mode", compile_active)
        self.assertEqual(
            (
                compile_active["cpu_clock_mhz_index"].low,
                compile_active["cpu_clock_mhz_index"].high,
            ),
            (0, 2),
        )
        self.assertNotIn("cpu_clock_mhz", compile_active)

        fixed_quantization = build_search_space_descriptor(
            family,
            context,
            {},
            runner_config(train=True, quantization_search=False),
            collect_compile_metrics=True,
        )
        self.assertNotIn("quantization_mode", fixed_quantization)

    def test_search_param_supports_continuous_float_sampling(self) -> None:
        """The common declaration supports the planned float parameter kind."""
        class Trial:
            def suggest_float(self, name, low, high):
                self.call = (name, low, high)
                return 0.25

        trial = Trial()
        param = SearchParam("learning_rate", "float", low=0.0, high=1.0)

        self.assertEqual(param.suggest(trial), 0.25)
        self.assertEqual(trial.call, ("learning_rate", 0.0, 1.0))

    def test_builtin_descriptors_expose_semantic_metadata(self) -> None:
        """Every built-in family parameter explains its architectural role."""
        odom = OdomTCNFamily().trial_search_space(odom_context(), {})
        self.assertEqual(
            {param.name for param in odom},
            {
                "dilations_index",
                "nb_filters",
                "kernel_size",
                "dropout_rate",
                "use_skip_connections",
                "norm_flag",
            },
        )
        self.assertTrue(all(param.description for param in odom))
        self.assertTrue(all(param.typical_effects for param in odom))

        audio_config = Dict(family="audio_dscnn", params=Dict(), search=Dict())
        audio = AudioDSCNNFamily().trial_search_space(audio_context(), audio_config)
        self.assertEqual({param.name for param in audio}, set(AUDIO_DSCNN_SEARCH_SEMANTICS))
        self.assertEqual(set(AUDIO_DSCNN_SEARCH_SEMANTICS), set(AUDIO_DSCNN_SEARCH_CHOICES))
        self.assertTrue(all(param.description for param in audio))
        self.assertTrue(all(param.typical_effects for param in audio))

    def test_third_party_descriptor_metadata_remains_optional(self) -> None:
        """Existing declarations remain valid without semantic fields."""
        param = SearchParam("legacy_width", "int", low=1, high=4)

        self.assertEqual(param.description, "")
        self.assertIsNone(param.units)
        self.assertEqual(param.typical_effects, ())


if __name__ == "__main__":
    unittest.main()
