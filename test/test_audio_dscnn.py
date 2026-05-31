"""Tests for the audio DS-CNN model family."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tensorflow as tf
from addict import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.model_families.audio_dscnn import (  # noqa: E402
    AUDIO_DSCNN_SEARCH_CHOICES,
    DEFAULT_AUDIO_DSCNN_SEED,
    AudioDSCNNFamily,
    resolve_stride_schedule,
)
from crest.pipeline_types import ModelBuildContext, TargetSpec  # noqa: E402

DEFAULT_TARGET_SENTINEL = object()


class DummyTrial:
    """Optuna-like trial stub that records categorical choice order."""

    def __init__(self) -> None:
        """Initialize the recorded choices and deterministic return values."""
        self.choices: dict[str, tuple[object, ...]] = {}

    def suggest_categorical(self, name, choices):
        """Record one categorical suggestion and return its last choice.

        Parameters
        ----------
        name : object
            Parameter name requested by the fake sampler.
        choices : object
            Candidate values available to the fake sampler.

        Returns
        -------
        object
            Selected categorical value from the fake sampler.
        """
        self.choices[name] = tuple(choices)
        return tuple(choices)[-1]


def make_target_spec(**metadata_overrides) -> TargetSpec:
    """Build the valid audio classification target spec for tests.

    Parameters
    ----------
    **metadata_overrides : object
        Optional metadata values to override on the valid baseline.

    Returns
    -------
    TargetSpec
        Classification target spec expected by ``AudioDSCNNFamily``.
    """
    metadata = {"num_classes": 10, "label_encoding": "class_index", "from_logits": True}
    metadata.update(metadata_overrides)
    return TargetSpec(
        task_type="classification",
        output_names=["class_logits"],
        output_shapes=[(10,)],
        metadata=metadata,
    )


def make_context(
    *,
    input_shape: tuple[int, ...] | None = (201, 64),
    target_spec: TargetSpec | None | object = DEFAULT_TARGET_SENTINEL,
) -> ModelBuildContext:
    """Build a valid model context with optional targeted overrides.

    Parameters
    ----------
    input_shape : tuple[int, ...] | None, optional
        Logical input shape to expose.
    target_spec : TargetSpec | None | object, optional
        Target spec to expose. Defaults to the valid audio contract; pass
        ``None`` to test missing-target behavior.

    Returns
    -------
    ModelBuildContext
        Build context used by model-family tests.
    """
    return ModelBuildContext(
        input_shape=input_shape,
        input_dtype="float32",
        target_spec=make_target_spec() if target_spec is DEFAULT_TARGET_SENTINEL else target_spec,
    )


def make_hparams(**overrides) -> dict[str, object]:
    """Build the default audio DS-CNN hyperparameter payload.

    Parameters
    ----------
    **overrides : object
        Hyperparameter values to override.

    Returns
    -------
    dict[str, object]
        Complete audio DS-CNN hyperparameter dictionary.
    """
    hparams = dict(DEFAULT_AUDIO_DSCNN_SEED)
    hparams.update(overrides)
    return hparams


def architecture_candidate_count() -> int:
    """Compute the architecture search-space size from family choices.

    Returns
    -------
    int
        Product of every model-family categorical choice count.
    """
    return math.prod(len(choices) for choices in AUDIO_DSCNN_SEARCH_CHOICES.values())


class AudioDSCNNFamilyTests(unittest.TestCase):
    """Validate the audio DS-CNN model-family contract."""

    def setUp(self) -> None:
        """Create the family and a valid model-build context."""
        self.family = AudioDSCNNFamily()
        self.ctx = make_context()
        self.config = Dict(family="audio_dscnn", params=Dict(), search=Dict())

    def tearDown(self) -> None:
        """Clear TensorFlow graph state after each test."""
        tf.keras.backend.clear_session()

    def test_family_constants_seed_and_search_size(self) -> None:
        """The family should expose the explicit DW/PW surface and seed."""
        self.assertEqual(self.family.name, "audio_dscnn")
        self.assertEqual(AUDIO_DSCNN_SEARCH_CHOICES["base_channels"], (4, 8, 12, 16, 20, 24, 32))
        self.assertEqual(AUDIO_DSCNN_SEARCH_CHOICES["num_blocks"], (2, 3, 4, 5, 6))
        self.assertEqual(
            AUDIO_DSCNN_SEARCH_CHOICES["stride_schedule"],
            ("light", "balanced", "aggressive", "early_time", "early_freq", "late"),
        )
        self.assertNotIn("depthwise_separable", AUDIO_DSCNN_SEARCH_CHOICES)
        self.assertEqual(architecture_candidate_count(), 7_741_440)
        self.assertEqual(architecture_candidate_count() * 5, 38_707_200)
        self.assertEqual(
            DEFAULT_AUDIO_DSCNN_SEED,
            {
                "base_channels": 16,
                "num_blocks": 3,
                "kernel_time": 3,
                "kernel_freq": 3,
                "stride_schedule": "balanced",
                "channel_growth": 2.0,
                "max_channels": 128,
                "depth_multiplier": 1,
                "pointwise_scale": 1.0,
                "dropout_rate": 0.1,
                "norm_flag": True,
                "dense_units": 32,
                "global_pool_type": "avg",
                "activation": "relu",
            },
        )
        self.assertEqual(self.family.default_seed_trial(self.ctx, self.config), DEFAULT_AUDIO_DSCNN_SEED)

    def test_sample_hparams_uses_override_order_for_new_fields(self) -> None:
        """Sampling should preserve override order and normalize float choices."""
        trial = DummyTrial()
        config = Dict(
            family="audio_dscnn",
            params=Dict(),
            search=Dict(
                base_channels=[24, 8],
                channel_growth=[2, 1.5],
                pointwise_scale=[1, 0.75],
                activation=["relu6", "relu"],
                global_pool_type=["max", "avg"],
            ),
        )

        sampled = self.family.sample_hparams(trial, self.ctx, config)

        self.assertEqual(trial.choices["base_channels"], (24, 8))
        self.assertEqual(trial.choices["channel_growth"], (2.0, 1.5))
        self.assertEqual(trial.choices["pointwise_scale"], (1.0, 0.75))
        self.assertEqual(trial.choices["activation"], ("relu6", "relu"))
        self.assertEqual(trial.choices["global_pool_type"], ("max", "avg"))
        self.assertEqual(sampled["base_channels"], 8)
        self.assertEqual(sampled["channel_growth"], 1.5)
        self.assertEqual(sampled["pointwise_scale"], 0.75)
        self.assertEqual(sampled["activation"], "relu")
        self.assertEqual(sampled["global_pool_type"], "avg")

    def test_empty_search_uses_family_default_choices(self) -> None:
        """An empty search block should use the full family default surface."""
        trial = DummyTrial()
        config = Dict(family="audio_dscnn", params=Dict(), search=Dict())

        self.family.sample_hparams(trial, self.ctx, config)

        for name, choices in AUDIO_DSCNN_SEARCH_CHOICES.items():
            self.assertEqual(trial.choices[name], choices)

    def test_validate_config_rejects_invalid_sections_and_legacy_keys(self) -> None:
        """Config validation should fail on unsupported params and search values."""
        valid = Dict(
            family="audio_dscnn",
            params=Dict(export_variant="trained_debug"),
            search=Dict(base_channels=[16, 8], channel_growth=[1, 1.5]),
        )
        self.family.validate_config(valid)

        invalid_cases = [
            Dict(params=Dict(), search=Dict()),
            Dict(family="odom_tcn", params=Dict(), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(other=True), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(export_variant=" "), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(unknown=[1])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(depthwise_separable=[True])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[8, 8])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(channel_growth=[1, 1.0])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[7])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(norm_flag=[1])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(pointwise_scale=[True])),
        ]
        for config in invalid_cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.family.validate_config(config)

    def test_validate_hparams_rejects_key_and_value_errors(self) -> None:
        """Hyperparameter validation should reject missing, extra, and bad values."""
        self.family.validate_hparams(make_hparams(dropout_rate=0), self.ctx, self.config)

        invalid_cases = [
            {key: value for key, value in make_hparams().items() if key != "kernel_time"},
            {**make_hparams(), "extra": True},
            {**make_hparams(), "depthwise_separable": True},
            make_hparams(base_channels=True),
            make_hparams(kernel_time=7),
            make_hparams(norm_flag=1),
            make_hparams(channel_growth=1.25),
            make_hparams(dropout_rate=0.25),
            make_hparams(global_pool_type="median"),
            make_hparams(activation="gelu"),
        ]
        for hparams in invalid_cases:
            with self.subTest(hparams=hparams):
                with self.assertRaises(ValueError):
                    self.family.validate_hparams(hparams, self.ctx, self.config)

    def test_validate_hparams_rejects_invalid_build_context(self) -> None:
        """Context validation should reject unsupported input and target contracts."""
        invalid_target_specs = [
            TargetSpec("regression", ["class_logits"], [(10,)], make_target_spec().metadata),
            TargetSpec("classification", ["other"], [(10,)], make_target_spec().metadata),
            TargetSpec("classification", ["class_logits"], [(9,)], make_target_spec().metadata),
            make_target_spec(num_classes=9),
            make_target_spec(from_logits=False),
            make_target_spec(label_encoding="one_hot"),
            TargetSpec("classification", ["class_logits"], [(10,)], {"num_classes": 10, "label_encoding": "class_index"}),
        ]

        for ctx in [
            make_context(input_shape=(201, 64, 1)),
            make_context(input_shape=(201.0, 64)),
            make_context(input_shape=("201", 64)),
            make_context(input_shape=(True, 64)),
            make_context(target_spec=None),
            make_context(target_spec=object()),
        ]:
            with self.subTest(ctx=ctx):
                with self.assertRaises(ValueError):
                    self.family.validate_hparams(make_hparams(), ctx, self.config)
        for target_spec in invalid_target_specs:
            with self.subTest(target_spec=target_spec):
                with self.assertRaises(ValueError):
                    self.family.validate_hparams(make_hparams(), make_context(target_spec=target_spec), self.config)

    def test_decode_trial_hparams_normalizes_storage_edge_cases(self) -> None:
        """Persisted trial decoding should normalize bool and float edge cases."""
        raw = make_hparams(norm_flag=0, channel_growth=2, pointwise_scale=1, dropout_rate=0)
        decoded = self.family.decode_trial_hparams(raw, self.ctx, self.config)

        self.assertIs(decoded["norm_flag"], False)
        self.assertEqual(decoded["channel_growth"], 2.0)
        self.assertEqual(decoded["pointwise_scale"], 1.0)
        self.assertEqual(decoded["dropout_rate"], 0.0)

        for raw_params in [
            make_hparams(norm_flag="0"),
            make_hparams(channel_growth="1.0"),
            make_hparams(pointwise_scale=True),
            make_hparams(dropout_rate="0.1"),
            {**make_hparams(), "depthwise_separable": True},
        ]:
            with self.subTest(raw_params=raw_params):
                with self.assertRaises(ValueError):
                    self.family.decode_trial_hparams(raw_params, self.ctx, self.config)

    def test_resolve_stride_schedule_outputs(self) -> None:
        """Stride schedules should resolve deterministically for variable depth."""
        self.assertEqual(resolve_stride_schedule("light", 2), ((1, 1), (2, 2)))
        self.assertEqual(resolve_stride_schedule("light", 5), ((1, 1), (2, 2), (1, 1), (1, 1), (1, 1)))
        self.assertEqual(resolve_stride_schedule("balanced", 2), ((2, 2), (1, 1)))
        self.assertEqual(resolve_stride_schedule("balanced", 4), ((2, 2), (1, 1), (2, 2), (1, 1)))
        self.assertEqual(resolve_stride_schedule("aggressive", 4), ((2, 2), (2, 2), (2, 2), (1, 1)))
        self.assertEqual(resolve_stride_schedule("early_time", 3), ((2, 1), (2, 1), (1, 1)))
        self.assertEqual(resolve_stride_schedule("early_freq", 3), ((1, 2), (1, 2), (1, 1)))
        self.assertEqual(resolve_stride_schedule("late", 3), ((1, 1), (1, 1), (2, 2)))
        self.assertEqual(resolve_stride_schedule("late", 5), ((1, 1), (1, 1), (2, 2), (1, 1), (2, 2)))

    def test_build_model_uses_audio_input_and_logits_output(self) -> None:
        """Model construction should produce one named linear logits output."""
        model = self.family.build_model(make_hparams(), self.ctx, self.config)

        self.assertEqual(model.input_shape, (None, 201, 64))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(model.output_names, ["class_logits"])
        self.assertEqual(model.layers[-1].activation.__name__, "linear")

    def test_depthwise_pointwise_layer_names_types_norm_and_bias(self) -> None:
        """Each block should materialize the explicit DW/PW layer sequence."""
        normalized = self.family.build_model(
            make_hparams(num_blocks=2, norm_flag=True, dropout_rate=0.1),
            self.ctx,
            self.config,
        )
        unnormalized = self.family.build_model(
            make_hparams(num_blocks=2, norm_flag=False, dense_units=0, dropout_rate=0.0),
            self.ctx,
            self.config,
        )

        expected_block_names = [
            "dscnn_block_1_depthwise_conv",
            "dscnn_block_1_depthwise_bn",
            "dscnn_block_1_depthwise_activation",
            "dscnn_block_1_pointwise_conv",
            "dscnn_block_1_pointwise_bn",
            "dscnn_block_1_pointwise_activation",
            "dscnn_block_1_dropout",
            "dscnn_block_2_depthwise_conv",
            "dscnn_block_2_depthwise_bn",
            "dscnn_block_2_depthwise_activation",
            "dscnn_block_2_pointwise_conv",
            "dscnn_block_2_pointwise_bn",
            "dscnn_block_2_pointwise_activation",
            "dscnn_block_2_dropout",
        ]
        for name in expected_block_names:
            with self.subTest(name=name):
                self.assertIsNotNone(normalized.get_layer(name))

        self.assertIsInstance(normalized.get_layer("dscnn_block_1_depthwise_conv"), tf.keras.layers.DepthwiseConv2D)
        self.assertIsInstance(normalized.get_layer("dscnn_block_1_pointwise_conv"), tf.keras.layers.Conv2D)
        self.assertFalse(normalized.get_layer("dscnn_block_1_depthwise_conv").use_bias)
        self.assertFalse(normalized.get_layer("dscnn_block_1_pointwise_conv").use_bias)
        self.assertEqual(
            sum(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in normalized.layers),
            4,
        )
        self.assertEqual(
            sum(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in unnormalized.layers),
            0,
        )
        self.assertTrue(unnormalized.get_layer("dscnn_block_1_depthwise_conv").use_bias)
        self.assertTrue(unnormalized.get_layer("dscnn_block_1_pointwise_conv").use_bias)
        self.assertFalse(any(layer.name == "dense_hidden" for layer in unnormalized.layers))

    def test_relu6_applies_to_blocks_and_dense_hidden(self) -> None:
        """The relu6 option should use ReLU layers capped at six."""
        model = self.family.build_model(
            make_hparams(num_blocks=2, activation="relu6", dense_units=16),
            self.ctx,
            self.config,
        )

        for name in [
            "dscnn_block_1_depthwise_activation",
            "dscnn_block_1_pointwise_activation",
            "dscnn_block_2_depthwise_activation",
            "dscnn_block_2_pointwise_activation",
            "dense_hidden_activation",
        ]:
            with self.subTest(name=name):
                layer = model.get_layer(name)
                self.assertIsInstance(layer, tf.keras.layers.ReLU)
                self.assertEqual(layer.max_value, 6.0)
        self.assertEqual(model.get_layer("dense_hidden").activation.__name__, "linear")

    def test_filter_formula_depth_multiplier_and_stride(self) -> None:
        """Pointwise filters, depth multiplier, and strides should follow hparams."""
        model = self.family.build_model(
            make_hparams(
                base_channels=12,
                num_blocks=4,
                channel_growth=1.5,
                pointwise_scale=1.25,
                max_channels=64,
                depth_multiplier=2,
                stride_schedule="aggressive",
            ),
            self.ctx,
            self.config,
        )

        self.assertEqual(model.get_layer("dscnn_block_1_pointwise_conv").filters, 15)
        self.assertEqual(model.get_layer("dscnn_block_2_pointwise_conv").filters, 22)
        self.assertEqual(model.get_layer("dscnn_block_3_pointwise_conv").filters, 34)
        self.assertEqual(model.get_layer("dscnn_block_4_pointwise_conv").filters, 50)
        self.assertEqual(model.get_layer("dscnn_block_1_depthwise_conv").depth_multiplier, 2)
        self.assertEqual(model.get_layer("dscnn_block_1_depthwise_conv").strides, (2, 2))
        self.assertEqual(model.get_layer("dscnn_block_4_depthwise_conv").strides, (1, 1))

        capped = self.family.build_model(
            make_hparams(base_channels=32, num_blocks=3, channel_growth=2.0, pointwise_scale=1.5, max_channels=64),
            self.ctx,
            self.config,
        )
        self.assertEqual(capped.get_layer("dscnn_block_1_pointwise_conv").filters, 48)
        self.assertEqual(capped.get_layer("dscnn_block_2_pointwise_conv").filters, 64)
        self.assertEqual(capped.get_layer("dscnn_block_3_pointwise_conv").filters, 64)

    def test_global_pooling_selection(self) -> None:
        """Average and max pooling choices should select matching layers."""
        avg_model = self.family.build_model(make_hparams(global_pool_type="avg"), self.ctx, self.config)
        max_model = self.family.build_model(make_hparams(global_pool_type="max"), self.ctx, self.config)

        self.assertIsInstance(avg_model.get_layer("global_avg_pool"), tf.keras.layers.GlobalAveragePooling2D)
        self.assertIsInstance(max_model.get_layer("global_max_pool"), tf.keras.layers.GlobalMaxPooling2D)

    def test_count_flops_returns_positive_for_small_seed_and_max_shape(self) -> None:
        """Small, seed, and maximum explicit DS-CNN shapes should have FLOPs."""
        ctx = make_context(input_shape=(16, 8))
        hparam_cases = [
            make_hparams(num_blocks=2, base_channels=4, dense_units=0, dropout_rate=0.0),
            make_hparams(num_blocks=3, dropout_rate=0.0),
            make_hparams(
                base_channels=32,
                num_blocks=6,
                kernel_time=5,
                kernel_freq=5,
                stride_schedule="late",
                channel_growth=2.0,
                max_channels=128,
                depth_multiplier=2,
                pointwise_scale=1.5,
                dropout_rate=0.0,
                dense_units=64,
                activation="relu6",
            ),
        ]

        for hparams in hparam_cases:
            with self.subTest(hparams=hparams):
                model = self.family.build_model(hparams, ctx, self.config)
                flops = self.family.count_flops(model, ctx, self.config)
                self.assertIsInstance(flops, int)
                self.assertGreater(flops, 0)

    def test_base_export_materialization_contract(self) -> None:
        """Base export materialization should remain the audio export contract."""
        hparams = make_hparams(num_blocks=2, dense_units=0, dropout_rate=0.0)
        untrained = self.family.materialize_export_model(
            hparams,
            self.ctx,
            self.config,
            model_variant="untrained",
        )
        self.assertEqual(untrained.output_shape, (None, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "trained.keras"
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            with patch.object(self.family, "load_model", return_value="loaded") as load_mock:
                loaded = self.family.materialize_export_model(
                    hparams,
                    self.ctx,
                    self.config,
                    model_variant="trained_debug",
                    checkpoint_path=checkpoint_path,
                )
        load_mock.assert_called_once_with(checkpoint_path, self.ctx, self.config)
        self.assertEqual(loaded, "loaded")

        with self.assertRaises(ValueError):
            self.family.materialize_export_model(hparams, self.ctx, self.config, model_variant="trained")
        with self.assertRaises(FileNotFoundError):
            self.family.materialize_export_model(
                hparams,
                self.ctx,
                self.config,
                model_variant="trained",
                checkpoint_path=Path("/tmp/audio-dscnn-missing.keras"),
            )
        with self.assertRaises(ValueError):
            self.family.materialize_export_model(hparams, self.ctx, self.config, model_variant="approx_trained")


if __name__ == "__main__":
    unittest.main()
