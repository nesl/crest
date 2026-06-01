# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the legacy OxIOD data-loading helpers."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest import data as crest_data

PREPARE_OXIOD_PATH = ROOT / "data" / "dataset_download_and_splits" / "oxiod" / "prepare_oxiod.py"
PREPARE_OXIOD_SPEC = importlib.util.spec_from_file_location("prepare_oxiod", PREPARE_OXIOD_PATH)
prepare_oxiod = importlib.util.module_from_spec(PREPARE_OXIOD_SPEC)
assert PREPARE_OXIOD_SPEC.loader is not None
PREPARE_OXIOD_SPEC.loader.exec_module(prepare_oxiod)


class FakeSlidingWindow:
    """Minimal sliding-window transform used to isolate loader behavior.

    Parameters
    ----------
    size : int
        Width of each extracted window.
    stride : int
        Step between consecutive windows.
    """

    def __init__(self, size, stride):
        """Initialize the instance.

        Parameters
        ----------
        size : object
            Window size used by the dataset helper.
        stride : object
            Window stride used by the dataset helper.
        """
        self.size = size
        self.stride = stride

    def fit_transform(self, signal):
        """Return overlapping windows using the same contract as the real helper.

        Parameters
        ----------
        signal : array-like
            One-dimensional signal to segment into windows.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_windows, size)``. Empty inputs return an empty
            array with the correct window width.
        """
        arr = np.asarray(signal).reshape(-1)
        windows = [arr[start:start + self.size] for start in range(0, arr.shape[0] - self.size + 1, self.stride)]
        if not windows:
            return np.empty((0, self.size))
        return np.stack(windows, axis=0)


def _identity_tqdm(iterable, *_, **__):
    """Return the iterable unchanged for deterministic, silent tests.

    Parameters
    ----------
    iterable : object
        Iterable wrapped by the helper.
    *_ : tuple[object, ...]
        Unused positional argument accepted by the test double.
    **__ : dict[str, object]
        Unused positional argument accepted by the test double.

    Returns
    -------
    object
        Iterable returned unchanged by the fake progress wrapper.
    """
    return iterable


class ImportOxIODDatasetMaxWindowsTests(unittest.TestCase):
    """Tests for OxIOD max-window import handling."""

    def setUp(self):
        # Patch the three moving pieces together so the loader exercises its own
        # window-capping logic without touching disk-heavy pandas/tqdm helpers.
        """Prepare test fixtures."""
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.tempdir.name)
        self.sub_folders = ["mock/", "mock_b/"]
        for folder in self.sub_folders:
            folder_path = self.dataset_root / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            (folder_path / "Train.txt").write_text("imu.csv\n")
        self.default_channels = [
            'Timestamp','Roll','Pitch','Yaw','Gyro_X','Gyro_Y','Gyro_Z','Grav_X','Grav_Y','Grav_Z',
            'Lin_Acc_X','Lin_Acc_Y','Lin_Acc_Z','Mag_X','Mag_Y','Mag_Z'
        ]
        self.default_gt_channels = ['Timestamp','Header','Pose_X','Pose_Y','Pose_Z','Rot_X','Rot_Y','Rot_Z','Rot_W']
        self.num_samples = 10
        self.window_size = 4
        self.stride = 2
        self.expected_windows = ((self.num_samples - self.window_size) // self.stride) + 1
        self.read_csv_patcher = patch('crest.data.pd.read_csv', side_effect=self.fake_read_csv)
        self.sliding_patcher = patch('crest.data.SlidingWindow', FakeSlidingWindow)
        self.tqdm_patcher = patch('crest.data.tqdm', _identity_tqdm)
        self.read_csv_patcher.start()
        self.sliding_patcher.start()
        self.tqdm_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        patch.stopall()
        self.tempdir.cleanup()

    def fake_read_csv(self, path, header=None):
        """Return deterministic IMU or VI frames keyed by the requested path.

        Parameters
        ----------
        path : object
            Path to the file used by the helper.
        header : object
            Header row supplied to the fake CSV reader.

        Returns
        -------
        object
            Data frame returned by the fake CSV reader.

        Raises
        ------
        FileNotFoundError
            If existing validation or execution checks fail.
        """
        if 'imu' in path:
            data = {col: np.arange(self.num_samples, dtype=float) for col in self.default_channels}
            return pd.DataFrame(data)
        if 'vi' in path:
            data = {col: np.arange(self.num_samples, dtype=float) for col in self.default_gt_channels}
            return pd.DataFrame(data)
        raise FileNotFoundError(path)

    def call_loader(self, max_windows=None, verbose=False):
        """Invoke ``import_oxiod_dataset`` with the shared test configuration.

        Parameters
        ----------
        max_windows : int | None, optional
            Optional cap forwarded to the loader under test.
        verbose : bool, optional
            Whether to enable progress logging in the loader.

        Returns
        -------
        object
            Legacy split object returned by ``import_oxiod_dataset``.
        """
        dataset_folder = str(self.dataset_root) + os.sep
        return crest_data.import_oxiod_dataset(
            type_flag=2,
            dataset_folder=dataset_folder,
            sub_folders=self.sub_folders,
            sampling_rate=100,
            window_size=self.window_size,
            stride=self.stride,
            verbose=verbose,
            useMagnetometer=False,
            useStepCounter=False,
            max_windows=max_windows,
        )

    def test_respects_max_windows_cap(self):
        # Split loading should honor the configured max-window cap when one is provided.
        """Validate respects max windows cap."""
        subset = self.call_loader(max_windows=2)
        self.assertEqual(subset.inputs.shape[0], 2)
        self.assertEqual(subset.size_of_each, [1, 1])
        self.assertEqual(subset.disp.shape[0], 2)
        self.assertEqual(subset.x_vel.shape[0], 2)

    def test_loads_full_split_without_cap(self):
        # Split loading should read the full split when no max-window cap is configured.
        """Validate loads full split without cap."""
        subset = self.call_loader(max_windows=None)
        total_expected = self.expected_windows * len(self.sub_folders)
        self.assertEqual(subset.inputs.shape[0], total_expected)
        self.assertEqual(subset.size_of_each, [self.expected_windows, self.expected_windows])
        self.assertEqual(subset.disp.shape[0], total_expected)
        self.assertEqual(subset.x_vel.shape[0], total_expected)

    def test_verbose_loader_logs_processing_message(self):
        """Verbose split loading should emit processing progress as logs.

        Returns
        -------
        None
            The test passes when verbose mode preserves the processing message
            through the module logger while returning the normal split data.
        """
        with self.assertLogs("crest.data", level="INFO") as captured:
            subset = self.call_loader(max_windows=1, verbose=True)

        self.assertEqual(subset.inputs.shape[0], 1)
        self.assertIn("Processing for (file and ground truth): mock/imu.csv", "\n".join(captured.output))


class PrepareOxIODTests(unittest.TestCase):
    """Validate the OxIOD dataset preparation helpers."""

    def test_capture_templates_reads_activity_split_files(self):
        """Capture_templates should read split files from activity folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir)
            activity = template_root / "handheld"
            activity.mkdir()
            (activity / "Train.txt").write_text("train.csv\n", encoding="utf-8")
            (activity / "Valid.txt").write_text("valid.csv\n", encoding="utf-8")

            templates = prepare_oxiod.capture_templates(template_root)

            self.assertEqual(templates[Path("handheld")]["Train.txt"], "train.csv\n")
            self.assertEqual(templates[Path("handheld")]["Valid.txt"], "valid.csv\n")

    def test_capture_templates_rejects_missing_templates(self):
        """Capture_templates should fail when no split files are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir)
            (template_root / "handheld").mkdir()

            with self.assertRaises(RuntimeError):
                prepare_oxiod.capture_templates(template_root)

    def test_normalize_folder_names_renames_slow_walking(self):
        """Normalize_folder_names should convert the OxIOD slow-walking folder name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_root = Path(tmpdir)
            original = dest_root / "slow walking"
            original.mkdir()

            prepare_oxiod.normalize_folder_names(dest_root)

            self.assertFalse(original.exists())
            self.assertTrue((dest_root / "slow_walking").is_dir())

    def test_restore_templates_writes_splits_to_activity_folders(self):
        """Restore_templates should write captured splits into destination activities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_root = Path(tmpdir)
            templates = {Path("handheld"): {"Train.txt": "imu.csv\n"}}

            prepare_oxiod.restore_templates(dest_root, templates)

            self.assertEqual((dest_root / "handheld" / "Train.txt").read_text(), "imu.csv\n")

    def test_main_uses_configured_paths_and_restores_templates(self):
        """Main should capture templates, extract, normalize, and restore splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "OxIOD.zip"
            zip_path.write_bytes(b"placeholder")
            template_root = root / "templates"
            activity = template_root / "handheld"
            activity.mkdir(parents=True)
            (activity / "Train.txt").write_text("imu.csv\n", encoding="utf-8")
            dest_root = root / "prepared"

            args = SimpleNamespace(
                zip_path=str(zip_path),
                dest_root=str(dest_root),
                template_root=str(template_root),
            )

            def fake_extract_archive(_zip_path: Path, requested_dest_root: Path) -> None:
                """Create a minimal extracted tree for main-path testing.

                Parameters
                ----------
                _zip_path : Path
                    Archive path intercepted by the download test double.
                requested_dest_root : Path
                    Destination root requested by the dataset download helper.
                """
                self.assertEqual(_zip_path, zip_path)
                requested_dest_root.mkdir(parents=True)

            with patch.object(prepare_oxiod, "parse_args", return_value=args), patch.object(
                prepare_oxiod, "extract_archive", side_effect=fake_extract_archive
            ):
                prepare_oxiod.main()

            self.assertEqual((dest_root / "handheld" / "Train.txt").read_text(), "imu.csv\n")


if __name__ == '__main__':
    unittest.main()
