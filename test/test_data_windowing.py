# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the OxIOD loader with real NumPy windowing and synthetic CSVs."""

from unittest.mock import patch

import numpy as np

from . import test_data as loader_tests
from crest import data as crest_data
from crest.windowing import SlidingWindow


class TestOxIODNumpyWindows(loader_tests.ImportOxIODDatasetMaxWindowsTests):
    """Reuse existing loader assertions without their windowing test double.

    Notes
    -----
    CSV reading and progress output retain the existing lightweight fixtures.
    The three inherited tests also run against the production window helper.
    """

    def setUp(self):
        """Prepare the existing fixtures and enable real window extraction.

        Returns
        -------
        None
            Registers the runtime patch for cleanup by the inherited fixture.
        """
        super().setUp()
        patch.object(crest_data, "SlidingWindow", SlidingWindow).start()

    def test_sensor_and_position_windows_share_end_alignment(self):
        """Keep sensor samples and displacement labels aligned at remainders.

        Returns
        -------
        None
            Asserts exact windows for a recording whose stride leaves a prefix.
        """
        self.num_samples = 11
        self.stride = 3
        subset = self.call_loader()
        samples = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
        samples = np.concatenate([samples, samples])
        np.testing.assert_array_equal(subset.inputs[:, :, :3],
                                      np.repeat((2 * samples)[:, :, None], 3, axis=2))
        np.testing.assert_array_equal(subset.inputs[:, :, 3:],
                                      np.repeat(samples[:, :, None], 3, axis=2))
        np.testing.assert_array_equal(subset.position,
                                      np.repeat(samples[:, :, None], 2, axis=2))
        np.testing.assert_array_equal(subset.x_vel, np.full(6, 3))
        np.testing.assert_array_equal(subset.y_vel, np.full(6, 3))
        np.testing.assert_allclose(subset.disp, np.full(6, np.sqrt(18)))
        self.assertEqual(subset.size_of_each, [3, 3])
