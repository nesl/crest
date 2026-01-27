import math
from dataclasses import dataclass
from typing import Optional

# import matplotlib.pyplot as plt
from math import atan2, sqrt  #, sin, cos, radians

import numpy as np
import pandas as pd
import quaternion
from gtda.time_series import SlidingWindow

# import os
from pydometer import Pedometer

# import h5py
# import json
# import glob
# from scipy.signal import savgol_filter
# from geographiclib.geodesic import Geodesic
from tqdm import tqdm

import geometry_helpers


@dataclass
class OxIODSplitData:
    """
    Container for the windowed OxIOD split returned by `import_oxiod_dataset`.

    Note: The original comment was as follows:
        ```
        # 1. training set from IMU 2. ground truth displacements 3. ground truth heading rates 4. ground truth position
        # 5. list of initial x positions 6. list of initial y positions 7. size of each file in windowed form
        # 8. ground truth x velocity 9. ground truth y velocity 10. heading rate in terms of sin 11. heading rate in terms of cos
        # 12. unwindowed training set from IMU
        ```
    
    Attributes
    ----------
    inputs : np.ndarray
        IMU (and optional step-counter) windows of shape
        `(n_windows, window_size, n_channels)`.
    disp : np.ndarray
        Absolute displacement magnitude per window, length `n_windows`.
    heading : np.ndarray
        Heading delta (degrees) per window, length `n_windows`.
    position : np.ndarray
        Ground-truth XY positions inside each window,
        shape `(n_windows, window_size, 2)`.
    x0 : list of float
        Initial X position of each raw trajectory.
    y0 : list of float
        Initial Y position of each raw trajectory.
    size_of_each : list of int
        Number of windows produced per trajectory.
    x_vel : np.ndarray
        Window-wise displacement along the X axis.
    y_vel : np.ndarray
        Window-wise displacement along the Y axis.
    head_s : np.ndarray
        Sine of the absolute heading for each window.
    head_c : np.ndarray
        Cosine of the absolute heading for each window.
    inputs_orig : np.ndarray
        Concatenated unwindowed IMU samples for all trajectories.
    """
    inputs: np.ndarray
    disp: np.ndarray
    heading: np.ndarray
    position: np.ndarray
    x0: list
    y0: list
    size_of_each: list
    x_vel: np.ndarray
    y_vel: np.ndarray
    head_s: np.ndarray
    head_c: np.ndarray
    inputs_orig: np.ndarray


def import_oxiod_dataset(type_flag = 2, useMagnetometer = True, useStepCounter = True, AugmentationCopies = 0,
                         dataset_folder = 'data/oxiod/',
                         sub_folders = ['handbag/','handheld/','pocket/','running/','slow_walking/','trolley/'],
                         sampling_rate = 100, window_size = 200, stride = 10, verbose=True,
                         max_windows: Optional[int] = None):
    """
    Load and window OxIOD IMU/ground-truth sequences according to split files.

    Parameters
    ----------
    type_flag : int, optional
        Split selector: 1=Train_Valid, 2=Train, 3=Valid, 4=Test, by default 2 (Train).
    useMagnetometer : bool, optional
        Include magnetometer channels when building feature tensors, by default True.
    useStepCounter : bool, optional
        Run a pedometer over accelerometer data and append the step indicator channel, by default True.
    AugmentationCopies : int, optional
        Number of random 3-D rotations applied per sample for augmentation, by default 0 (no augmentation).
    dataset_folder : str, optional
        Base directory containing dataset subfolders; should end with a slash, by default 'oxiod/'.
    sub_folders : list of str, optional
        Relative subfolder paths searched for split files (e.g., ['handbag/', 'handheld/', ...]),
          by default all six OxIOD motion types.
    sampling_rate : int, optional
        IMU sampling frequency in Hz, by default 100.
    window_size : int, optional
        Sliding window size in samples, by default 200.
    stride : int, optional
        Sliding window stride in samples, by default 10.
    verbose : bool, optional
        Print filenames as they are processed when True, by default False.
    max_windows : int | None, optional
        Hard cap on the number of sliding windows to load. When set, the loader
        stops once ``max_windows`` examples have been added. This is useful for
        lightweight calibration/quantization passes that only need a small,
        representative subset. ``None`` (default) loads the full split.

    Returns
    -------
    OxIODSplitData
        Dataclass containing windowed IMU inputs and ground-truth data.

    Notes
    -----
    Expects split files (Train.txt, Valid.txt, etc.) inside each subfolder,
    listing IMU CSV paths whose paired ground-truth files are obtained via
    ``path.replace('imu', 'vi')``.
    """
    default_channels = ['Timestamp','Roll','Pitch','Yaw','Gyro_X','Gyro_Y','Gyro_Z','Grav_X','Grav_Y','Grav_Z','Lin_Acc_X','Lin_Acc_Y','Lin_Acc_Z','Mag_X','Mag_Y','Mag_Z']
    default_GT_channels = ['Timestamp','Header','Pose_X','Pose_Y','Pose_Z','Rot_X','Rot_Y','Rot_Z','Rot_W']
    wanted_GT_channels = ['Pose_X','Pose_Y','Pose_Z']
    if(type_flag == 1): #full training set (including validation)
        type_file = 'Train_Valid.txt'
    elif(type_flag==2): #training set
        type_file = 'Train.txt'
    elif(type_flag==3): #validation set
        type_file = 'Valid.txt'
    elif(type_flag==4): #test set
        type_file = 'Test.txt'
    else:
        raise ValueError("type_flag must be 1, 2, 3, or 4")
    
    if(useMagnetometer):
        channel_count = 9
    else:
        channel_count = 6
    
    X_orig = np.empty([0,channel_count])
    x0_list = []
    y0_list = []
    size_of_each = []
    X = np.empty([0, window_size, channel_count])
    Y_disp = np.empty([0])
    Y_head = np.empty([0])
    Y_pos = np.empty([0,window_size, 2])
    x_vel = np.empty([0])
    y_vel = np.empty([0])
    head_s = np.empty([0])
    head_c = np.empty([0])
    
    if(useStepCounter):
        loc_3D_mat = np.empty([0,window_size])
       
    total_windows = 0
    budget_reached = False
    remaining_folders = len(sub_folders)

    for folder in tqdm(sub_folders):
        if budget_reached:
            break
        folder_windows = 0
        folder_budget = None
        if max_windows is not None:
            remaining_budget = max_windows - total_windows
            if remaining_budget <= 0:
                budget_reached = True
                break
            folder_budget = math.ceil(remaining_budget / remaining_folders)
        with open(dataset_folder+folder+type_file, 'r') as f:
            list_of_files = [line.strip() for line in f]
        for line in list_of_files:
            if budget_reached:
                break
            if folder_budget is not None and folder_windows >= folder_budget:
                break
            if(verbose==True):
                print('Processing for (file and ground truth): '+folder+line)
            cur_train = pd.read_csv(dataset_folder+folder+line,header=None)
            cur_train.columns = default_channels
            acc_x = cur_train['Lin_Acc_X'].to_numpy() + cur_train['Grav_X'].to_numpy()
            acc_y = cur_train['Lin_Acc_Y'].to_numpy() + cur_train['Grav_Y'].to_numpy()
            acc_z = cur_train['Lin_Acc_Z'].to_numpy() + cur_train['Grav_Z'].to_numpy()
            gyr_x = cur_train['Gyro_X'].to_numpy().reshape((acc_x.shape[0],1))
            gyr_y = cur_train['Gyro_Y'].to_numpy().reshape((acc_x.shape[0],1))
            gyr_z = cur_train['Gyro_Z'].to_numpy().reshape((acc_x.shape[0],1))
            
            if(useMagnetometer):
                mag_x = cur_train['Mag_X'].to_numpy().reshape((acc_x.shape[0],1))
                mag_y = cur_train['Mag_Y'].to_numpy().reshape((acc_x.shape[0],1))
                mag_z = cur_train['Mag_Z'].to_numpy().reshape((acc_x.shape[0],1)) 
                
            if(useStepCounter):
                df = pd.DataFrame({'gx': acc_x, 'gy': acc_y, 'gz': acc_z})
                p = Pedometer(data=df, sr=sampling_rate)
                step_count, step_locations = p.get_steps()
                loc = np.zeros(cur_train.shape[0])
                loc[step_locations] = 1
           
            acc_x = acc_x.reshape((acc_x.shape[0],1))
            acc_y = acc_y.reshape((acc_y.shape[0],1))
            acc_z = acc_z.reshape((acc_z.shape[0],1))
           
            if(useMagnetometer):
                cur_train = np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,mag_x,mag_y,mag_z),axis=1)
            else:
                cur_train = np.concatenate((acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z),axis=1)
                                           
            cur_GT = pd.read_csv(dataset_folder+folder+line.replace('imu','vi'),header=None)
            cur_GT.columns = default_GT_channels
            cur_GT.drop(list(set(default_GT_channels) - set(wanted_GT_channels)), axis=1,inplace=True)
            cur_GT = cur_GT.to_numpy()
   
            windows = SlidingWindow(size=window_size, stride=stride)
            cur_train_3D = windows.fit_transform(cur_train[:,0])
            for i in range(1,cur_train.shape[1]):
                X_windows = windows.fit_transform(cur_train[:,i])
                cur_train_3D = np.dstack((cur_train_3D,X_windows))
            
            if(useStepCounter):   
                loc_3D = windows.fit_transform(loc)

            cur_GT_3D = windows.fit_transform(cur_GT[:,0])
            for i in range(1,cur_GT.shape[1]):
                X_windows = windows.fit_transform(cur_GT[:,i])
                cur_GT_3D = np.dstack((cur_GT_3D,X_windows))  
           
            vx = np.zeros((cur_GT_3D.shape[0]))
            vy = np.zeros((cur_GT_3D.shape[0]))
            
            heading_s = np.zeros((cur_GT_3D.shape[0]))
            heading_c = np.zeros((cur_GT_3D.shape[0]))
            for i in range(cur_GT_3D.shape[0]):
                s,c = abs_heading_sin_cos(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
                heading_s[i] = s
                heading_c[i] = c            
           
            displacement_GT_abs = np.zeros(cur_GT_3D.shape[0])
            heading_GT = np.zeros((cur_GT_3D.shape[0]))
            prev = 0
            for i in range(cur_GT_3D.shape[0]):
                Xdisp = (cur_GT_3D[i,-1,0]-cur_GT_3D[i,0,0])
                vx[i] = Xdisp
                Ydisp = (cur_GT_3D[i,-1,1]-cur_GT_3D[i,0,1])
                vy[i] = Ydisp
                displacement_GT_abs[i] = sqrt((Xdisp**2) + (Ydisp**2))  
                theta = abs_heading(cur_GT_3D[i,-1,0],cur_GT_3D[i,-1,1],cur_GT_3D[i,0,0],cur_GT_3D[i,0,1])
                if theta<180:
                    theta = theta + 180
       
                heading_GT[i] = theta - prev
                if(heading_GT[i]>100 or heading_GT[i]<-100):
                    theta2 = theta
                    prev2 = prev
                    if theta<prev:
                        theta2 = theta + 360
                    else:
                        prev2 =  prev + 360
                    heading_GT[i] = theta2 - prev2
                prev = theta
            
            if max_windows is not None:
                global_remaining = max_windows - total_windows
                folder_remaining = folder_budget - folder_windows if folder_budget is not None else global_remaining
                remaining = min(global_remaining, folder_remaining)
                if remaining <= 0:
                    budget_reached = True
                    break
                if cur_train_3D.shape[0] > remaining:
                    cur_train_3D = cur_train_3D[:remaining]
                    cur_GT_3D = cur_GT_3D[:remaining]
                    displacement_GT_abs = displacement_GT_abs[:remaining]
                    heading_GT = heading_GT[:remaining]
                    vx = vx[:remaining]
                    vy = vy[:remaining]
                    heading_s = heading_s[:remaining]
                    heading_c = heading_c[:remaining]
                    if(useStepCounter):
                        loc_3D = loc_3D[:remaining]


            current_windows = cur_train_3D.shape[0]

            X = np.vstack((X, cur_train_3D))
            X_orig = np.concatenate((X_orig,cur_train))
            Y_disp = np.concatenate((Y_disp, displacement_GT_abs))
            Y_head = np.concatenate((Y_head, heading_GT))
            Y_pos = np.vstack((Y_pos, cur_GT_3D[:,:,0:2]))
            x0_list.append(cur_GT[0,0])
            y0_list.append(cur_GT[0,1])
            size_of_each.append(current_windows)
            x_vel = np.concatenate((x_vel, vx))
            y_vel = np.concatenate((y_vel, vy))
            head_s = np.concatenate((head_s,heading_s))
            head_c = np.concatenate((head_c,heading_c))
            if(useStepCounter):
                loc_3D_mat = np.vstack((loc_3D_mat,loc_3D))
            total_windows += current_windows
            folder_windows += current_windows
            if max_windows is not None and total_windows >= max_windows:
                budget_reached = True
                break
            if(AugmentationCopies>0):
                for i in range(AugmentationCopies):
                    if max_windows is not None and total_windows >= max_windows:
                        budget_reached = True
                        break
                    out = random_rotate(cur_train_3D, useMagnetometer)
                    if max_windows is not None:
                        global_remaining = max_windows - total_windows
                        folder_remaining = folder_budget - folder_windows if folder_budget is not None else global_remaining
                        remaining = min(global_remaining, folder_remaining)
                        if remaining <= 0:
                            budget_reached = True
                            break
                        if out.shape[0] > remaining:
                            out = out[:remaining]
                    aug_len = out.shape[0]
                    X = np.vstack((X, out))
                    X_orig = np.concatenate((X_orig,cur_train))
                    Y_disp = np.concatenate((Y_disp, displacement_GT_abs[:aug_len]))
                    Y_head = np.concatenate((Y_head, heading_GT[:aug_len]))
                    Y_pos = np.vstack((Y_pos, cur_GT_3D[:aug_len,:,0:2]))
                    x0_list.append(cur_GT[0,0])
                    y0_list.append(cur_GT[0,1])
                    size_of_each.append(aug_len)
                    x_vel = np.concatenate((x_vel, vx[:aug_len]))
                    y_vel = np.concatenate((y_vel, vy[:aug_len]))
                    head_s = np.concatenate((head_s,heading_s[:aug_len]))
                    head_c = np.concatenate((head_c,heading_c[:aug_len]))
                    if(useStepCounter):
                        loc_3D_mat = np.vstack((loc_3D_mat,loc_3D[:aug_len]))
                    total_windows += aug_len
                    folder_windows += aug_len
                if budget_reached:
                    break
        remaining_folders -= 1
           
    if(useStepCounter):
        X = np.concatenate((X,loc_3D_mat.reshape(loc_3D_mat.shape[0],loc_3D_mat.shape[1],1)),axis=2)
    
    # returns 1. training set from IMU 2. ground truth displacements 3. ground truth heading rates 4. ground truth position
    # 5. list of initial x positions 6. list of initial y positions 7. size of each file in windowed form
    # 8. ground truth x velocity 9. ground truth y velocity 10. heading rate in terms of sin 11. heading rate in terms of cos
    # 12. unwindowed training set from IMU
    return  OxIODSplitData(inputs=X, 
                           disp=Y_disp, 
                           heading=Y_head, 
                           position=Y_pos, 
                           x0=x0_list,
                           y0=y0_list, 
                           size_of_each=size_of_each, 
                           x_vel=x_vel, 
                           y_vel=y_vel, 
                           head_s=head_s, 
                           head_c=head_c, 
                           inputs_orig=X_orig)

def abs_heading(cur_x, cur_y, prev_x, prev_y):
        dely = (cur_y - prev_y)
        delx = (cur_x - prev_x)
        delh= atan2(delx,dely)*57.2958
        return delh
    
def abs_heading_sin_cos(cur_x, cur_y, prev_x, prev_y, eps=1e-4):
    """
    Compute the sine and cosine of the absolute heading between two 2D points.

    Given current and previous (x, y) positions, this function calculates the
    normalized direction of motion as sine (`s`) and cosine (`c`) components.
    If the displacement magnitude is below a small threshold (`eps`), the
    heading is considered undefined and `(0.0, 0.0)` is returned instead of
    performing an unstable division.

    Parameters
    ----------
    cur_x : float or np.ndarray
        Current x-coordinate(s) of the trajectory point(s).
    cur_y : float or np.ndarray
        Current y-coordinate(s) of the trajectory point(s).
    prev_x : float or np.ndarray
        Previous x-coordinate(s) of the trajectory point(s).
    prev_y : float or np.ndarray
        Previous y-coordinate(s) of the trajectory point(s).
    eps : float, optional
        Minimum displacement (in meters) to be considered as motion.
        Displacements smaller than `eps` are treated as zero to prevent
        division-by-zero errors. Default is 1e-4 (≈ 0.1 mm).

    Returns
    -------
    s : float or np.ndarray
        The sine component of the absolute heading (Δy / √(Δx² + Δy²)).
        Returns 0.0 if the displacement is below `eps`.
    c : float or np.ndarray
        The cosine component of the absolute heading (Δx / √(Δx² + Δy²)).
        Returns 0.0 if the displacement is below `eps`.

    Notes
    -----
    - The function assumes coordinates are expressed in meters, as in the
      OxIOD dataset (≈0.5 mm positional accuracy).
    - When there is negligible motion (`sqr < eps`), `(s, c) = (0.0, 0.0)`
      indicates a stationary window without a meaningful heading.
    - For temporal consistency, callers may choose to propagate the previous
      heading instead of returning zeros.

    Examples
    --------
    >>> abs_heading_sin_cos(0.5, 1.0, 0.4, 0.8)
    (0.8944271909999159, 0.4472135954999579)

    >>> abs_heading_sin_cos(0.0, 0.0, 0.0, 0.0)
    (0.0, 0.0)
    """
    dely = cur_y - prev_y
    delx = cur_x - prev_x
    sqr = np.sqrt(dely * dely + delx * delx)
    if sqr < eps:
        return 0.0, 0.0
    try:
        s = dely / sqr
        c = delx / sqr
    except RuntimeWarning:
        s = 0.0
        c = 0.0

        raise Exception(f"Warning: Division by zero encountered in abs_heading_sin_cos, sqr={sqr}, dely={dely}, delx={delx}.")
    return s, c


def random_rotate(input,useMagnetometer=True):
    output = np.copy(input)
    euler = np.random.uniform(0, np.pi, size=3)
    for i in range(0, input.shape[0]):
        input_acc = input[i,:,0:3]
        input_rot = input[i,:,3:6]
        if(useMagnetometer):
            input_mag = input[i,:,6:9]  
        Rot = geometry_helpers.euler2mat(euler[0],euler[1],euler[2])
        output_acc = np.dot(Rot, input_acc.T).T
        output_rot = np.dot(Rot, input_rot.T).T
        if(useMagnetometer):
            output_mag = np.dot(Rot, input_mag.T).T
            output[i,:,:] = np.hstack((output_acc, output_rot, output_mag))  
        else:
            output[i,:,:] = np.hstack((output_acc, output_rot))    
    return output

def orientation_to_angles(ori):
    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles


def adjust_angle_array(angles):
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def Cal_TE(Gvx, Gvy, Pvx, Pvy, sampling_rate=100, window_size=200, stride=10, length=None):
    """Compute average and relative translation errors between ground-truth and predicted 2D trajectories.

    This function compares ground-truth positions ``(Gvx, Gvy)`` with predicted
    positions ``(Pvx, Pvy)`` and returns the Average Translation Error (ATE)
    over the full trajectory, as well as an estimated Relative Translation
    Error (RTE) over a one-minute horizon based on the provided sampling and
    windowing parameters.

    Parameters
    ----------
    Gvx : array-like of float
        Ground-truth x-coordinates of the trajectory, in meters.
    Gvy : array-like of float
        Ground-truth y-coordinates of the trajectory, in meters.
    Pvx : array-like of float
        Predicted x-coordinates of the trajectory, in meters.
    Pvy : array-like of float
        Predicted y-coordinates of the trajectory, in meters.
    sampling_rate : int, optional
        Sampling frequency in Hz (samples per second). Default is ``100``.
    window_size : int, optional
        Sliding-window size in samples used to define a one-minute horizon.
        Default is ``200``.
    stride : int, optional
        Sliding-window stride in samples used for the one-minute horizon.
        Default is ``10``.
    length : int or None, optional
        Number of samples to use from the beginning of the sequences.
        If ``None``, all samples in ``Gvx`` are used. Default is ``None``.

    Returns
    -------
    ate : float
        Average Translation Error over the full trajectory, computed as the
        mean Euclidean distance between ground-truth and predicted positions.
    rte : float
        Estimated Relative Translation Error over a one-minute horizon.
        If the trajectory is shorter than one minute, ``rte`` is obtained by
        scaling ``ate`` by ``n_windows_one_min / length``.
    at_all : list of float
        Per-sample Euclidean position errors for the full trajectory.
    rt_all : list of float
        Per-sample Euclidean position errors used for the one-minute RTE
        computation (typically the first ``n_windows_one_min`` samples).
    """

    if length is None:
        length = len(Gvx)

    distance = []

    for i in range(length):
        d = ((Gvx[i] - Pvx[i]) * (Gvx[i] - Pvx[i])) + ((Gvy[i] - Pvy[i]) * (Gvy[i] - Pvy[i]))
        d = math.sqrt(d)
        distance.append(d)

    mean_distance = sum(distance) / len(distance)
    ate = mean_distance
    at_all = distance

    n_windows_one_min = int(((sampling_rate * 60) - window_size) / stride)
    distance = []
    if n_windows_one_min < length:
        for i in range(n_windows_one_min):
            d = ((Gvx[i] - Pvx[i]) * (Gvx[i] - Pvx[i])) + ((Gvy[i] - Pvy[i]) * (Gvy[i] - Pvy[i]))
            d = math.sqrt(d)
            distance.append(d)
        rte = sum(distance) / len(distance)
    else:
        rte = ate * (n_windows_one_min / length)

    rt_all = distance
    return ate, rte, at_all, rt_all


def Cal_len_meters(Gvx, Gvy, length=None):
    """Compute total path length from 2D ground-truth positions.

    The function treats successive pairs of positions ``(Gvx[i], Gvy[i])`` and
    ``(Gvx[i-1], Gvy[i-1])`` as straight-line segments and sums their
    Euclidean distances to obtain the total traveled distance.

    Parameters
    ----------
    Gvx : array-like of float
        Ground-truth x-coordinates of the trajectory, in meters.
    Gvy : array-like of float
        Ground-truth y-coordinates of the trajectory, in meters.
    length : int or None, optional
        Number of samples to use from the beginning of the sequences.
        If ``None``, all samples in ``Gvx`` are used. Default is ``None``.

    Returns
    -------
    float
        Total path length in meters, computed as the sum of Euclidean
        distances between consecutive ground-truth samples.
    """

    if length is None:
        length = len(Gvx)

    distance = []

    for i in range(1, length):
        d = ((Gvx[i] - Gvx[i - 1]) * (Gvx[i] - Gvx[i - 1])) + ((Gvy[i] - Gvy[i - 1]) * (Gvy[i] - Gvy[i - 1]))
        d = math.sqrt(d)
        distance.append(d)

    sum_distance = sum(distance)

    return sum_distance
