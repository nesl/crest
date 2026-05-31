"""Describe oxIOD dataset adapter built on top of the legacy split loader.

This module wraps :func:`import_oxiod_dataset`, normalizes the returned legacy
splits into :class:`DatasetBundle`, and optionally loads a capped calibration
subset when representative-data limits are configured.
"""

from __future__ import annotations

from typing import Any

from ..data import import_oxiod_dataset
from ..interfaces import DatasetABC
from ..pipeline_types import DataSplit, DatasetBundle

_OXIOD_SUBFOLDERS = [
    "handbag/",
    "handheld/",
    "pocket/",
    "running/",
    "slow_walking/",
    "trolley/",
]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Return one configuration field from a mapping- or attribute-like object.

    Parameters
    ----------
    config : Any
        Configuration object that may expose ``get`` or attributes.
    key : str
        Field name to retrieve.
    default : Any, optional
        Fallback value when the field is missing.

    Returns
    -------
    Any
        Retrieved configuration value or ``default``.
    """
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


class OxIODDataset(DatasetABC):
    """Dataset adapter that exposes the current OxIOD loader via ``DatasetABC``.

    The adapter translates legacy OxIOD split payloads into generic
    ``DatasetBundle`` / ``DataSplit`` objects, preserves compatibility metadata
    needed by later phases, and implements the current calibration fallback
    behavior used by the modular pipeline.
    """

    @property
    def name(self) -> str:
        """Return the stable registry-facing dataset name.

        Returns
        -------
        str
            Stable dataset identifier.
        """
        return "oxiod"

    def validate_config(self, dataset_config: Any) -> None:
        """Validate the OxIOD dataset configuration required by Phase 2.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree. The current adapter requires
            ``directory``, ``sampling_rate_hz``, ``window_size``, and
            ``stride``. ``calibration_windows`` is optional but must be
            positive when provided.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a required field is missing or numerically invalid.
        """
        required_keys = ("directory", "sampling_rate_hz", "window_size", "stride")
        missing = [key for key in required_keys if _cfg_get(dataset_config, key) in (None, "")]
        if missing:
            raise ValueError(f"OxIODDataset requires config values for: {', '.join(missing)}.")

        for key in ("sampling_rate_hz", "window_size", "stride"):
            value = _cfg_get(dataset_config, key)
            if int(value) <= 0:
                raise ValueError(f"OxIODDataset config '{key}' must be a positive integer.")

        calibration_windows = _cfg_get(dataset_config, "calibration_windows", None)
        if calibration_windows is not None and int(calibration_windows) <= 0:
            raise ValueError("OxIODDataset config 'calibration_windows' must be positive when set.")

    def load(self, dataset_config: Any) -> DatasetBundle:
        """Load the current CREST OxIOD splits into a generic dataset bundle.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree. The current adapter consumes
            ``directory``, ``sampling_rate_hz``, ``window_size``, ``stride``,
            and optional ``calibration_windows``.

        Returns
        -------
        DatasetBundle
            Generic bundle containing the current CREST train, validation,
            test, and optional calibration splits. Train/validation/test are
            loaded through the legacy split selector, with train/validation
            explicitly enabling magnetometer and pedometer features while the
            test split preserves the legacy default call shape by omitting
            those kwargs. When ``calibration_windows`` is configured, the
            adapter performs one additional capped training-split load to
            populate ``bundle.calibration``. Bundle metadata currently includes
            ``sampling_rate_hz``, ``window_size``, ``stride``, and
            ``input_dim``.
        """
        self.validate_config(dataset_config)
        directory = str(_cfg_get(dataset_config, "directory"))
        sampling_rate_hz = int(_cfg_get(dataset_config, "sampling_rate_hz"))
        window_size = int(_cfg_get(dataset_config, "window_size"))
        stride = int(_cfg_get(dataset_config, "stride"))
        calibration_windows = _cfg_get(dataset_config, "calibration_windows", None)

        common_kwargs = {
            "dataset_folder": directory,
            "sub_folders": list(_OXIOD_SUBFOLDERS),
            "sampling_rate": sampling_rate_hz,
            "window_size": window_size,
            "stride": stride,
            "verbose": False,
        }

        train_legacy = import_oxiod_dataset(
            type_flag=2,
            useMagnetometer=True,
            useStepCounter=True,
            AugmentationCopies=0,
            **common_kwargs,
        )
        val_legacy = import_oxiod_dataset(
            type_flag=3,
            useMagnetometer=True,
            useStepCounter=True,
            AugmentationCopies=0,
            **common_kwargs,
        )
        # Preserve the current NASModelClient test-split call shape, which
        # relies on the legacy loader defaults for magnetometer and pedometer
        # handling rather than spelling them out again.
        test_legacy = import_oxiod_dataset(type_flag=4, **common_kwargs)

        calibration = None
        if calibration_windows is not None:
            calibration_legacy = import_oxiod_dataset(
                type_flag=2,
                useMagnetometer=True,
                useStepCounter=True,
                AugmentationCopies=0,
                max_windows=int(calibration_windows),
                **common_kwargs,
            )
            calibration = self._to_split(calibration_legacy)

        train = self._to_split(train_legacy)
        return DatasetBundle(
            train=train,
            val=self._to_split(val_legacy),
            test=self._to_split(test_legacy),
            calibration=calibration,
            input_shape=tuple(train_legacy.inputs.shape[1:]),
            input_dtype=str(train_legacy.inputs.dtype),
            metadata={
                "sampling_rate_hz": sampling_rate_hz,
                "window_size": window_size,
                "stride": stride,
                "input_dim": int(train_legacy.inputs.shape[2]),
            },
        )

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: Any,
    ) -> DataSplit | None:
        """Return calibration data with legacy OxIOD fallback semantics.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package produced by :meth:`load`.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DataSplit | None
            Explicitly capped calibration data when configured, otherwise the
            full training split to preserve the legacy ``max_windows=None``
            representative-data behavior. When no cap is configured, the
            fallback intentionally returns ``bundle.train``.
        """
        calibration_windows = _cfg_get(dataset_config, "calibration_windows", None)
        if calibration_windows is None:
            return bundle.train
        return bundle.calibration

    def _to_split(self, legacy_split: Any) -> DataSplit:
        """Translate one legacy OxIOD split into a generic ``DataSplit``.

        Parameters
        ----------
        legacy_split : Any
            Split payload returned by ``import_oxiod_dataset``.

        Returns
        -------
        DataSplit
            Generic split with ``velx`` / ``vely`` targets and compatibility
            metadata retaining legacy trajectory and evaluation fields for
            later phases.
        """
        return DataSplit(
            inputs=legacy_split.inputs,
            targets={"velx": legacy_split.x_vel, "vely": legacy_split.y_vel},
            sample_weights=None,
            metadata={
                "disp": legacy_split.disp,
                "heading": legacy_split.heading,
                "position": legacy_split.position,
                "x0": legacy_split.x0,
                "y0": legacy_split.y0,
                "size_of_each": legacy_split.size_of_each,
                "head_s": legacy_split.head_s,
                "head_c": legacy_split.head_c,
                "inputs_orig": legacy_split.inputs_orig,
            },
        )
