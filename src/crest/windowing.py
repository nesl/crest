# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Extract end-aligned time-series windows using NumPy."""

import numpy as np


class SlidingWindow:
    """Extract independent windows ending at the last input sample.

    Parameters
    ----------
    size : int, default=10
        Number of consecutive samples in each window.
    stride : int, default=1
        Distance between successive window starts.

    Notes
    -----
    Preserves the window alignment and writable-copy behavior of the former
    giotto-tda helper for CREST's numeric inputs. Only ``fit_transform`` is
    provided; this is not a scikit-learn estimator. When the stride does not
    divide ``n_samples - size``, discard the leading remainder so the final
    sample is always included. No padding is applied.
    """

    def __init__(self, size=10, stride=1):
        """Store the window parameters, validated when transforming data.

        Parameters
        ----------
        size : int, default=10
            Window width in samples.
        stride : int, default=1
            Distance between successive window starts.
        """
        self.size = size
        self.stride = stride

    def fit_transform(self, signal):
        """Return end-aligned windows without fitting any learned state.

        Parameters
        ----------
        signal : array-like of shape (n_samples, ...)
            Finite real-valued samples. Axis zero is time. Numeric object
            arrays are converted to float64; other numeric dtypes are kept.

        Returns
        -------
        numpy.ndarray of shape (n_windows, size, ...)
            Independent, writable windows in chronological order, where
            ``n_windows = (n_samples - size) // stride + 1``. Neither the
            input nor overlapping windows share storage with one another.

        Raises
        ------
        TypeError
            If size or stride is not a Python integer, or signal is scalar.
        ValueError
            If size or stride is nonpositive, the input is empty or shorter
            than size, or input values are nonnumeric, complex, or nonfinite.
        """
        for name, value in (("size", self.size), ("stride", self.stride)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        data = np.asarray(signal)
        if data.ndim == 0:
            raise TypeError("Expected a time series, not a scalar")
        if data.size == 0:
            raise ValueError("Expected a nonempty time series")
        if data.dtype.kind == "O":
            data = data.astype(np.float64)
        if data.dtype.kind not in "biuf":
            raise ValueError("Expected real numeric input")
        if not np.isfinite(data).all():
            raise ValueError("Expected finite input")
        if len(data) < self.size:
            raise ValueError("Time series is shorter than the window")

        offset = (len(data) - self.size) % self.stride
        view = np.lib.stride_tricks.sliding_window_view(
            data, self.size, axis=0
        )[offset::self.stride]
        # NumPy appends the window axis; callers expect (windows, time, ...).
        return np.moveaxis(view, -1, 1).copy()
