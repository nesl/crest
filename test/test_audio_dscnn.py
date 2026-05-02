"""Tests for the audio DS-CNN model family."""

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

from tinyodom.model_families.audio_dscnn import (  # noqa: E402
    AUDIO_DSCNN_SEARCH_CHOICES,
    DEFAULT_AUDIO_DSCNN_SEED,
    STRIDE_SCHEDULES,
    AudioDSCNNFamily,
)
from tinyodom.pipeline_types import ModelBuildContext, TargetSpec  # noqa: E402

DEFAULT_TARGET_SENTINEL = object()


class DummyTrial:
    """Optuna-like trial stub that records categorical choice order."""

    def __init__(self) -> None:
        """Initialize the recorded choices and deterministic return values."""

        self.choices: dict[str, tuple[object, ...]] = {}

    def suggest_categorical(self, name, choices):
        """Record one categorical suggestion and return its last choice."""

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


class AudioDSCNNFamilyTests(unittest.TestCase):
    """Validate the Phase 4 audio DS-CNN model-family contract."""

    def setUp(self) -> None:
        """Create the family and a valid model-build context."""

        self.family = AudioDSCNNFamily()
        self.ctx = make_context()
        self.config = Dict(family="audio_dscnn", params=Dict(), search=Dict())

    def tearDown(self) -> None:
        """Clear TensorFlow graph state after each test."""

        tf.keras.backend.clear_session()

    def test_family_name_constants_and_seed_trial(self) -> None:
        """The family should expose stable names, constants, and seed params."""

        self.assertEqual(self.family.name, "audio_dscnn")
        self.assertEqual(AUDIO_DSCNN_SEARCH_CHOICES["base_channels"], (8, 12, 16, 24, 32))
        self.assertEqual(STRIDE_SCHEDULES["balanced"], ((2, 2), (1, 1), (2, 2), (1, 1)))
        self.assertEqual(self.family.default_seed_trial(self.ctx, self.config), DEFAULT_AUDIO_DSCNN_SEED)

    def test_sample_hparams_uses_override_order(self) -> None:
        """Sampling should preserve config override order for Optuna choices."""

        trial = DummyTrial()
        config = Dict(
            family="audio_dscnn",
            params=Dict(),
            search=Dict(
                base_channels=[24, 8],
                global_pool_type=["max", "avg"],
            ),
        )

        sampled = self.family.sample_hparams(trial, self.ctx, config)

        self.assertEqual(trial.choices["base_channels"], (24, 8))
        self.assertEqual(trial.choices["global_pool_type"], ("max", "avg"))
        self.assertEqual(sampled["base_channels"], 8)
        self.assertEqual(sampled["global_pool_type"], "avg")

    def test_validate_config_rejects_invalid_sections(self) -> None:
        """Config validation should fail on unsupported params and search values."""

        valid = Dict(family="audio_dscnn", params=Dict(export_variant="trained_debug"), search=Dict(base_channels=[16, 8]))
        self.family.validate_config(valid)

        invalid_cases = [
            Dict(params=Dict(), search=Dict()),
            Dict(family="odom_tcn", params=Dict(), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(other=True), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(export_variant=" "), search=Dict()),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(unknown=[1])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[8, 8])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(base_channels=[7])),
            Dict(family="audio_dscnn", params=Dict(), search=Dict(depthwise_separable=[1])),
        ]
        for config in invalid_cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.family.validate_config(config)

    def test_validate_hparams_rejects_key_and_value_errors(self) -> None:
        """Hyperparameter validation should reject missing, extra, and bad values."""

        self.family.validate_hparams(make_hparams(), self.ctx, self.config)

        invalid_cases = [
            {key: value for key, value in make_hparams().items() if key != "kernel_time"},
            {**make_hparams(), "extra": True},
            make_hparams(base_channels=True),
            make_hparams(kernel_time=7),
            make_hparams(depthwise_separable=1),
            make_hparams(dropout_rate=0),
            make_hparams(dropout_rate=0.25),
            make_hparams(global_pool_type="median"),
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

        raw = make_hparams(depthwise_separable=1, norm_flag=0, dropout_rate=0)
        decoded = self.family.decode_trial_hparams(raw, self.ctx, self.config)

        self.assertIs(decoded["depthwise_separable"], True)
        self.assertIs(decoded["norm_flag"], False)
        self.assertEqual(decoded["dropout_rate"], 0.0)

        for raw_params in [
            make_hparams(depthwise_separable="true"),
            make_hparams(norm_flag="0"),
            make_hparams(dropout_rate="0.1"),
        ]:
            with self.subTest(raw_params=raw_params):
                with self.assertRaises(ValueError):
                    self.family.decode_trial_hparams(raw_params, self.ctx, self.config)

    def test_build_model_uses_audio_input_and_logits_output(self) -> None:
        """Model construction should produce one named logits output."""

        model = self.family.build_model(make_hparams(), self.ctx, self.config)

        self.assertEqual(model.input_shape, (None, 201, 64))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(model.output_names, ["class_logits"])
        self.assertEqual(model.layers[-1].activation.__name__, "linear")

    def test_conv_family_norm_bias_and_dense_options(self) -> None:
        """Architecture options should select the expected Keras layers."""

        separable = self.family.build_model(make_hparams(depthwise_separable=True), self.ctx, self.config)
        standard = self.family.build_model(
            make_hparams(depthwise_separable=False, norm_flag=False, dense_units=0),
            self.ctx,
            self.config,
        )

        self.assertIsInstance(separable.get_layer("dscnn_block_1_conv"), tf.keras.layers.SeparableConv2D)
        self.assertFalse(separable.get_layer("dscnn_block_1_conv").use_bias)
        self.assertTrue(any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in separable.layers))
        self.assertIsInstance(standard.get_layer("dscnn_block_1_conv"), tf.keras.layers.Conv2D)
        self.assertTrue(standard.get_layer("dscnn_block_1_conv").use_bias)
        self.assertFalse(any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in standard.layers))
        self.assertFalse(any(layer.name == "dense_hidden" for layer in standard.layers))

    def test_filter_cap_and_stride_schedule_truncation(self) -> None:
        """Filter growth should cap at 128 and truncate schedules by block count."""

        model = self.family.build_model(
            make_hparams(base_channels=32, num_blocks=4, stride_schedule="aggressive"),
            self.ctx,
            self.config,
        )

        self.assertEqual(model.get_layer("dscnn_block_4_conv").filters, 128)
        self.assertEqual(model.get_layer("dscnn_block_1_conv").strides, (2, 2))
        self.assertEqual(model.get_layer("dscnn_block_4_conv").strides, (1, 1))

    def test_count_flops_returns_positive_integer(self) -> None:
        """The frozen-graph FLOP counter should return a positive integer."""

        ctx = make_context(input_shape=(16, 8))
        hparams = make_hparams(num_blocks=2, base_channels=8, dense_units=0, dropout_rate=0.0)
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
