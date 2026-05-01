"""Shared UrbanSound8K constants for the audio backend.

Phase 3 dataset-loader code must import class names and cache constants from
this module instead of redefining them. Keeping the mapping in one location
prevents label/logit ordering drift between preparation, loading, and scoring.
"""

from __future__ import annotations

CACHE_VERSION = "v1_logmel_16k_2s_64mels_25ms_10ms"
COMPONENT_NAME = "urbansound8k_mel"
DATASET_NAME = "urbansound8k"
FEATURE_KIND = "log_mel_power"
FEATURE_DTYPE = "float32"
LABEL_ENCODING = "class_index"
LOG_FLOOR = 1e-6
NORMALIZATION_EPSILON = 1e-8

SAMPLE_RATE_HZ = 16000
CLIP_DURATION_S = 2.0
TARGET_NUM_SAMPLES = 32000
MEL_BINS = 64
WINDOW_MS = 25
WINDOW_LENGTH_SAMPLES = 400
HOP_MS = 10
HOP_LENGTH_SAMPLES = 160
N_FFT = 512
CENTER = True
POWER = 2.0
EXPECTED_FRAMES = 201
BATCH_PERIOD_MS = 2000
TRAIN_CROP_VARIANTS = 3
CROP_SEED = 1337
CALIBRATION_EXAMPLES = 100

CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)

FOLD_SPLIT = {
    "train": (1, 2, 3, 4, 5, 6, 7, 8),
    "val": (9,),
    "test": (10,),
}

REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "dataset_name",
    "component_name",
    "cache_version",
    "feature_kind",
    "sample_rate_hz",
    "clip_duration_s",
    "target_num_samples",
    "mel_bins",
    "window_ms",
    "window_length_samples",
    "hop_ms",
    "hop_length_samples",
    "n_fft",
    "center",
    "power",
    "log_floor",
    "log_scale",
    "expected_frames",
    "feature_dtype",
    "normalization",
    "normalization_mean_shape",
    "normalization_std_shape",
    "normalization_mean",
    "normalization_std",
    "normalization_epsilon",
    "fold_split",
    "batch_period_ms",
    "train_crop_variants",
    "crop_seed",
    "calibration_examples",
    "class_names",
)

REQUIRED_CACHE_ARRAYS = (
    "inputs",
    "labels",
    "folds",
    "clip_ids",
    "crop_start_sample",
    "crop_variant_index",
    "pad_before_sample",
    "pad_after_sample",
)
