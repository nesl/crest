# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for NumPy windowing without a giotto dependency."""

import numpy as np
import pytest

from crest.windowing import SlidingWindow


@pytest.mark.parametrize("length,size,stride,expected", [
    (10, 4, 4, [[2, 3, 4, 5], [6, 7, 8, 9]]),
    (10, 3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    (5, 5, 2, [[0, 1, 2, 3, 4]]),
    (6, 1, 2, [[1], [3], [5]]),
    (10, 2, 5, [[3, 4], [8, 9]]),
    (4, 2, 10, [[2, 3]]),
])
def test_known_window_contents(length, size, stride, expected):
    """Verify explicit examples characterized against giotto-tda 0.6.2.

    Parameters
    ----------
    length : int
        Number of input samples.
    size : int
        Window width.
    stride : int
        Window step.
    expected : list of list of int
        Independently specified window contents.
    """
    np.testing.assert_array_equal(
        SlidingWindow(size, stride).fit_transform(np.arange(length)), expected
    )


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(17,), (17, 2), (17, 2, 3)])
@pytest.mark.parametrize("reverse", [False, True])
def test_shape_dtype_and_noncontiguous_input(dtype, shape, reverse):
    """Preserve values and axes for noncontiguous multichannel inputs.

    Parameters
    ----------
    dtype : numpy.dtype
        Input element type.
    shape : tuple of int
        Shape after subsampling.
    reverse : bool
        Whether to reverse the time axis.
    """
    signal = np.arange(np.prod(shape) * 2, dtype=dtype).reshape(
        (shape[0] * 2,) + shape[1:]
    )[::2]
    if reverse:
        signal = signal[::-1]
    expected = np.stack([signal[1:5], signal[4:8], signal[7:11],
                         signal[10:14], signal[13:17]])
    actual = SlidingWindow(4, 3).fit_transform(signal)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype


def test_randomized_windows_from_recording_end():
    """Compare seeded signals with backward slicing from the final sample.

    Notes
    -----
    Uses ordinary slices in reverse order as an independent reference, not
    the production implementation's stride-trick or remainder calculation.
    """
    rng = np.random.default_rng(20261123)
    for _ in range(100):
        length = int(rng.integers(2, 500))
        size = int(rng.integers(1, length + 1))
        stride = int(rng.integers(1, length + 10))
        signal = rng.normal(size=(length, 6))
        expected = np.stack([
            signal[end - size:end]
            for end in range(length, size - 1, -stride)
        ][::-1])
        np.testing.assert_array_equal(
            SlidingWindow(size, stride).fit_transform(signal), expected
        )


@pytest.mark.parametrize("signal,error", [
    ([], ValueError), (1, TypeError), ([1, np.nan, 3], ValueError),
    ([1, np.inf, 3], ValueError), ([1, -np.inf, 3], ValueError),
    ([1j, 2j], ValueError), (["1", "2"], ValueError),
])
def test_invalid_data_rejected(signal, error):
    """Reject malformed input with the characterized exception type.

    Parameters
    ----------
    signal : object
        Invalid input sample collection.
    error : type of Exception
        Expected exception type.
    """
    with pytest.raises(error):
        SlidingWindow(1, 1).fit_transform(signal)


@pytest.mark.parametrize("size,stride,error", [
    (0, 1, ValueError), (-1, 1, ValueError), (1, 0, ValueError),
    (1, -1, ValueError), (1.5, 1, TypeError), (1, 1.5, TypeError),
    ("2", 1, TypeError), (1, None, TypeError),
])
def test_invalid_parameters_rejected(size, stride, error):
    """Reject nonpositive or noninteger window parameters.

    Parameters
    ----------
    size : object
        Requested window width.
    stride : object
        Requested step.
    error : type of Exception
        Expected exception type.
    """
    with pytest.raises(error):
        SlidingWindow(size, stride).fit_transform(np.arange(10))


def test_short_recording_rejected():
    """Reject insufficient samples rather than pad or return empty windows.

    Returns
    -------
    None
        Raises an assertion failure if a short recording is accepted.
    """
    with pytest.raises(ValueError):
        SlidingWindow(4, 20).fit_transform(np.arange(3))


def test_windows_are_independent_writable_copies():
    """Ensure changing one window cannot mutate input or another window.

    Returns
    -------
    None
        Raises an assertion failure if storage is shared.
    """
    signal = np.arange(6)
    windows = SlidingWindow(3, 1).fit_transform(signal)
    windows[0, 1] = 999
    assert windows[1, 0] == 1
    assert signal[1] == 1
    assert not np.shares_memory(signal, windows)


def test_numeric_object_input():
    """Preserve the legacy conversion of numeric objects to float64.

    Returns
    -------
    None
        Raises an assertion failure if conversion or values differ.
    """
    actual = SlidingWindow(2, 1).fit_transform(np.array([1, 2, 3], dtype=object))
    np.testing.assert_array_equal(actual, [[1, 2], [2, 3]])
    assert actual.dtype == np.float64
