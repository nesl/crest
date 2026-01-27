# OxIOD Dataset Download and Splitting Guide:

- Download the dataset from here: http://deepio.cs.ox.ac.uk/
- Rename the zip file to OxIOD.zip
- Run Dataset setup below

## Dataset setup

To create the splits, use the prepare_oxiod.py script. 
Ex: `python data/dataset_download_and_splits/prepare_oxiod.py --zip-path OxIOD.zip`.

The script unpacks the archive into `data/oxiod`, normalizes folder names (e.g., `slow walking` → `slow_walking`), and restores the curated `Train.txt`, `Valid.txt`, `Test.txt`, and `Train_Valid.txt` files that are already tracked in the repo for each activity.

## Dataset details:
- Within each folder in the dataset (in our case: ```handbag```, ```handheld```, ```pocket```, ```running```, ```slow_walking``` and ```trolley```), put files similar to the ```.txt``` files provided in this folder. They refer to which IMU files (and ground truth files to import). The ```.txt``` files we gave are just examples. You have to create splits through them for each folder.
- Check the ```data_utils.py``` file to see how TinyOdom imports data.
- Note: While the test/train/valid text files are included in this repo, the data files are not to limit the repo size

## Raw vs Syn Data

Each data folder contains `raw/` and `syn/` subfolders:
- **`raw/`**: Raw, unsynchronized data with high-precision timestamps. IMU and VI measurements may not be time-aligned.
- **`syn/`**: Synchronized data where IMU and VI are aligned, but with slightly less precise timestamps.

For odometry tasks, **use `syn/`** (as done in the provided splits) since synchronization is crucial for accurate IMU-ground-truth pairing. If you need ultra-precise timing, switch to `raw/` and handle sync manually.

## Subfolder Split Details

The splits use `syn/` data and mix files across data folders for representativeness. `Train_Valid` combines Train and Valid for full training (used when `type_flag=1` in `data_utils.py`). Note the train/validate/test files were shuffled prior to being put into these categories.

| Subfolder    | Total Files | Train          | Valid         | Test          | Train_Valid    |
|--------------|-------------|----------------|---------------|---------------|----------------|
| handbag      | 8           | 5 (62.5%)     | 2 (25%)       | 1 (12.5%)     | 7 (87.5%)      |
| handheld     | 24          | 18 (75%)      | 4 (16.7%)     | 2 (8.3%)      | 22 (91.7%)     |
| pocket       | 11          | 7 (63.6%)     | 3 (27.3%)     | 1 (9.1%)      | 10 (90.9%)     |
| running      | 7           | 5 (71.4%)     | 1 (14.3%)     | 1 (14.3%)     | 6 (85.7%)      |
| slow walking | 8           | 5 (62.5%)     | 2 (25%)       | 1 (12.5%)     | 7 (87.5%)      |
| trolley      | 13          | 10 (76.9%)    | 2 (15.4%)     | 1 (7.7%)      | 12 (92.3%)     |
| **Total**    | **71**      | **50 (70.4%)**| **14 (19.7%)**| **7 (9.9%)**  | **64 (90.1%)** |
