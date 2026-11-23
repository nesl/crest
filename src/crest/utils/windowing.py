# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Extract end-aligned time-series windows using NumPy."""

from numbers import Integral

import numpy as np


class SlidingWindow:
    """Extract independent windows ending at the last input sample.

    Parameters
    ----------
    size : int or numpy.integer, default=10
        Number of consecutive samples in each window.
    stride : int or numpy.integer, default=1
        Distance between successive window starts.

    Notes
    -----
    Preserves the window alignment and writable-copy behavior of the former
    giotto-tda helper for CREST's numeric inputs. Only ``fit_transform`` is
    provided; this is not a scikit-learn estimator. When the stride does not
    divide ``n_samples - size``, discard the leading remainder so the final
    sample is always included. No padding is applied.

    Python and NumPy integers are accepted for size and stride; booleans are
    rejected explicitly. This extends the former helper's Python-integer
    parameter contract without changing window contents for valid inputs.
    """

    def __init__(self, size=10, stride=1):
        """Store the window parameters, validated when transforming data.

        Parameters
        ----------
        size : int or numpy.integer, default=10
            Window width in samples.
        stride : int or numpy.integer, default=1
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
            If size or stride is not a Python or NumPy integer, is a boolean,
            or signal is scalar.
        ValueError
            If size or stride is nonpositive, the input is empty or shorter
            than size, or input values are nonnumeric, complex, or nonfinite.
        """
        for name, value in (("size", self.size), ("stride", self.stride)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, not a boolean")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        size, stride = int(self.size), int(self.stride)
        data = np.asarray(signal)
        if data.ndim == 0:
            raise TypeError("Expected a time series, not a scalar")
        if data.size == 0:
            raise ValueError("Expected a nonempty time series")
        if data.dtype.kind == "O":
            if any(isinstance(value, (complex, np.complexfloating)) for value in data.flat):
                raise ValueError("Expected real numeric input")
            try:
                data = data.astype(np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Expected real numeric input convertible to float64") from exc
        if data.dtype.kind not in "biuf":
            raise ValueError("Expected real numeric input")
        if not np.isfinite(data).all():
            raise ValueError("Expected finite input")
        if len(data) < size:
            raise ValueError("Time series is shorter than the window")

        offset = (len(data) - size) % stride
        view = np.lib.stride_tricks.sliding_window_view(
            data, size, axis=0
        )[offset::stride]
        # NumPy appends the window axis; callers expect (windows, time, ...).
        return np.moveaxis(view, -1, 1).copy()
