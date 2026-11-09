# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Dataset loading and calibration-data contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..pipeline_types import DataSplit, DatasetBundle


class DatasetABC(ABC):
    """Abstract dataset contract for modular pipeline implementations.

    Implementations must provide :meth:`load` and may optionally override the
    default hooks for config validation and calibration-data selection. By
    default, :attr:`name` returns the implementing class name,
    :meth:`validate_config` performs no validation, and
    :meth:`make_calibration_data` forwards ``bundle.calibration`` unchanged.
    """

    @property
    def name(self) -> str:
        """Return a human-readable dataset identifier.

        Returns
        -------
        str
            Default identifier derived from the implementing class name.
        """
        return type(self).__name__

    @abstractmethod
    def load(self, dataset_config: Any) -> DatasetBundle:
        """Load, preprocess, and normalize a dataset bundle.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DatasetBundle
            Normalized dataset package ready for downstream pipeline stages.
            The returned bundle is also the source object consumed by later
            dataset hooks, including the default calibration-data passthrough.
        """
    def validate_config(self, dataset_config: Any) -> None:
        """Validate dataset-local configuration.

        Parameters
        ----------
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        None
            The default implementation intentionally performs no validation and
            accepts any config object.
        """
        del dataset_config

    def make_calibration_data(
        self,
        bundle: DatasetBundle,
        dataset_config: Any,
    ) -> DataSplit | None:
        """Return calibration data used by export-time representative sampling.

        Parameters
        ----------
        bundle : DatasetBundle
            Normalized dataset package produced by :meth:`load`.
        dataset_config : Any
            Dataset-local configuration subtree.

        Returns
        -------
        DataSplit | None
            Calibration split when one is available. The default implementation
            ignores ``dataset_config``, forwards ``bundle.calibration``
            unchanged, and may therefore return ``None``.
        """
        del dataset_config
        return bundle.calibration
