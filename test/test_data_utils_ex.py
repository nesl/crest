import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_utils_ex


class FakeSlidingWindow:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def fit_transform(self, signal):
        arr = np.asarray(signal).reshape(-1)
        windows = [arr[start:start + self.size] for start in range(0, arr.shape[0] - self.size + 1, self.stride)]
        if not windows:
            return np.empty((0, self.size))
        return np.stack(windows, axis=0)


def _identity_tqdm(iterable, *_, **__):
    return iterable


class ImportOxIODDatasetMaxWindowsTests(unittest.TestCase):
    def setUp(self):
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
        self.read_csv_patcher = patch('src.data_utils_ex.pd.read_csv', side_effect=self.fake_read_csv)
        self.sliding_patcher = patch('src.data_utils_ex.SlidingWindow', FakeSlidingWindow)
        self.tqdm_patcher = patch('src.data_utils_ex.tqdm', _identity_tqdm)
        self.read_csv_patcher.start()
        self.sliding_patcher.start()
        self.tqdm_patcher.start()

    def tearDown(self):
        patch.stopall()
        self.tempdir.cleanup()

    def fake_read_csv(self, path, header=None):
        if 'imu' in path:
            data = {col: np.arange(self.num_samples, dtype=float) for col in self.default_channels}
            return pd.DataFrame(data)
        if 'vi' in path:
            data = {col: np.arange(self.num_samples, dtype=float) for col in self.default_gt_channels}
            return pd.DataFrame(data)
        raise FileNotFoundError(path)

    def call_loader(self, max_windows=None):
        dataset_folder = str(self.dataset_root) + os.sep
        return data_utils_ex.import_oxiod_dataset(
            type_flag=2,
            dataset_folder=dataset_folder,
            sub_folders=self.sub_folders,
            sampling_rate=100,
            window_size=self.window_size,
            stride=self.stride,
            verbose=False,
            useMagnetometer=False,
            useStepCounter=False,
            max_windows=max_windows,
        )

    def test_respects_max_windows_cap(self):
        subset = self.call_loader(max_windows=2)
        self.assertEqual(subset.inputs.shape[0], 2)
        self.assertEqual(subset.size_of_each, [1, 1])
        self.assertEqual(subset.disp.shape[0], 2)
        self.assertEqual(subset.x_vel.shape[0], 2)

    def test_loads_full_split_without_cap(self):
        subset = self.call_loader(max_windows=None)
        total_expected = self.expected_windows * len(self.sub_folders)
        self.assertEqual(subset.inputs.shape[0], total_expected)
        self.assertEqual(subset.size_of_each, [self.expected_windows, self.expected_windows])
        self.assertEqual(subset.disp.shape[0], total_expected)
        self.assertEqual(subset.x_vel.shape[0], total_expected)


if __name__ == '__main__':
    unittest.main()
