import os
import numpy as np
import sys
from numpy import float32, sin, cos, arctan, tan
import math
import datetime
from datetime import datetime
from concurrent import futures
import matplotlib.pyplot as plt

# to get the undistorted points data in first point coordinate
# and save all points to files so that mapping can use it.
NSCANS = 64
scanPeriod = 0.1  # velodyne scanperiod\appro 0.1s

POINT_LIST = []
for i in range(3):
    file = "{:010}.txt".format(i)
    POINT_LIST.append(file)


IMU_LIST = []
for i in range(2967):
    file = "{:010}.txt".format(i)
    IMU_LIST.append(file)


DIR_NAME = "E:/kitti_raw_data/unrectified/"
DATA = os.path.join(DIR_NAME, "2011_09_30_drive_0016_extract")

TEST_DIR = os.path.join(DATA, "velodyne_points")
TEST_FILES = [os.path.join(TEST_DIR, 'data', file) for file in POINT_LIST]

IMU_DIR = os.path.join(DATA, 'oxts')
IMU_FILES = [os.path.join(IMU_DIR, 'data', file) for file in IMU_LIST]


if not os.path.exists(os.path.join(TEST_DIR, "data_modified_math")): os.mkdir(os.path.join(TEST_DIR, "data_modified_math"))
SAVE_DIR = os.path.join(TEST_DIR, "data_modified_math")
SAVE_DIR = os.path.join(TEST_DIR, "data_modified_math_test")


def R_from_RPY(roll, pitch, yaw):
    R_roll = np.array([[cos(roll), -sin(roll), 0],
                       [sin(roll), cos(roll), 0],
                       [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0],
                        [0, cos(pitch), sin(pitch)],
                        [0, -sin(pitch), cos(pitch)]])

    R_yaw = np.array([[cos(yaw), 0, -sin(yaw)],
                      [0, 1, 0],
                      [sin(yaw), 0, cos(yaw)]])
    return R_yaw @ R_pitch @ R_roll


def R_from_YPR(yaw, pitch, roll):
    R_roll = np.array([[cos(roll), -sin(roll), 0],
                       [sin(roll), cos(roll), 0],
                       [0, 0, 1]])

    R_pitch = np.array([[1, 0, 0],
                        [0, cos(pitch), sin(pitch)],
                        [0, -sin(pitch), cos(pitch)]])

    R_yaw = np.array([[cos(yaw), 0, -sin(yaw)],
                      [0, 1, 0],
                      [sin(yaw), 0, cos(yaw)]])
    return R_yaw @ R_pitch @ R_roll


def getTimeList(timeStamps):
    with open(timeStamps, 'r') as file:
        time_list = file.readlines()
        time_list_ = [datetime.timestamp(datetime.strptime(time[:-4], '%Y-%m-%d %H:%M:%S.%f')) for time in time_list]
    return time_list_


def getImuData(imu_files):
    """
    @imu_files List[str]\n
    imu_data are all relative to the World Coordinate system\n
    the velocity and shift needed to be calculated by hand\n
    RPY are the Euler angles relative the the world coordinate system\n
    a vector v in world coordinate is v = Y * P * R * v', where v' is the velodyne and imu coordinate system\n
    because these two systems are the same.\n
    return: ndarray shape N x 30, where N is the time stamps and 30 is the number of imu data
    """
    return np.asarray([np.loadtxt(os.path.join(IMU_DIR, 'data', imu_file), dtype=np.float32) for imu_file in imu_files])


def getPoints(test_file):
    """
    get one stamp points  \n
    @return N x 4 ndarray
    """
    print("Handle {} data--------------------".format(test_file))
    return np.loadtxt(test_file, dtype=np.float32)


def getPointsFromBin(test_file):
    print("Handle {} data--------------------".format(test_file))
    return np.fromfile(test_file, dtype=np.float32).reshape(-1, 4)


def imuHandle(imu_data, imu_time_stamp):
    """
    @ imu_data: imu information ndarray len(Stamps) x 30\n
    @ imu_time_stamp: imu_time stamp List[datetime.datetime]\n
    @ return: len(stamps) x 30 imu_data
    remove the gravity acceleration impact and calculate velocity and shift since first imu_stamp\n
    First remove g = 9.81m/s^2\n
    And then  calculate shift and velocity from 0 point\n
    a little change to oxtdata: imudata[24:27] are imushitX/Y/Z info and imu[27:30] are imuvelocityX/Y/Z
    """
    size = imu_data.shape[0]  ## N -> the number of stamps
    for i in range(size):

        roll, pitch, yaw = imu_data[i, 3], imu_data[i, 4], imu_data[i, 5]
        acceleration_x, acceleration_y, acceleration_z = imu_data[i, 11], imu_data[i, 12], imu_data[i, 13]
        # remove the gravity
        # rotate to z-forward x-left y-up
        # because RPY is such coordinate
        acc_x = acceleration_y - np.sin(roll) * np.cos(pitch) * 9.81
        acc_y = acceleration_z - np.cos(roll) * np.cos(pitch) * 9.81
        acc_z = acceleration_x + np.sin(pitch) * 9.81
        imu_data[i, 11], imu_data[i, 12], imu_data[i, 13] = acc_x, acc_y, acc_z

        # rotate to world coordinate
        acc_xyz_imu = np.array([acc_x, acc_y, acc_z])
        [acc_x, acc_y, acc_z] = R_from_RPY(roll, pitch, yaw) @ acc_xyz_imu

        # now we know that all accelerations are in world coordinate system and can be used to remove distortion
        # when using these data, we need to transform all velodyne system points to world coordinate use imu RPY
        # now calculate the shifts and velocities use accelerations but not built-in velocities
        # the last three entries([27:30]) are useless, so use these entries to hold shift infomation
        # use [8:11] to hold velocity infomation
        # these data are all world coordinate information
        
        if i == 0:
            # if the first point, assume that all infomation are stored since here, and initial values are all 0
            imu_data[i, 27], imu_data[i, 28], imu_data[i, 29] = 0.0, 0.0, 0.0
            roll_start, pitch_start, yaw_start = imu_data[i, 3], imu_data[i, 4], imu_data[i, 5]
            [imu_data[i, 8], imu_data[i, 9], imu_data[i, 10]] = R_from_RPY(roll_start, pitch_start, yaw_start) @ [imu_data[i, 8], imu_data[i, 9], imu_data[i, 10]]
        else:
            roll_start, pitch_start, yaw_start = imu_data[i, 3], imu_data[i, 4], imu_data[i, 5]
            [imu_data[i, 8], imu_data[i, 9], imu_data[i, 10]] = R_from_RPY(roll_start, pitch_start, yaw_start) @ [imu_data[i, 8], imu_data[i, 9], imu_data[i, 10]]

            time_diff = imu_time_stamp[i] - imu_time_stamp[i - 1]

            if time_diff < scanPeriod:
                imu_data[i, 27] = imu_data[i - 1, 27] + imu_data[
                    i - 1, 8] * time_diff + 0.5 * acc_x * time_diff * time_diff
                imu_data[i, 28] = imu_data[i - 1, 28] + imu_data[
                    i - 1, 9] * time_diff + 0.5 * acc_y * time_diff * time_diff
                imu_data[i, 29] = imu_data[i - 1, 29] + imu_data[
                    i - 1, 10] * time_diff + 0.5 * acc_z * time_diff * time_diff

                '''
                imu_data[i, 8] = imu_data[i - 1, 8] + acc_x * time_diff
                imu_data[i, 9] = imu_data[i - 1, 9] + acc_y * time_diff
                imu_data[i, 10] = imu_data[i - 1, 10] + acc_z * time_diff
                '''

    return imu_data


def getLastAndCur(imu_time_stamp, point_time_stamp):
    """
    @ imu_time_stamp: the list of imu_time_stamp\n
    @ point_time_stamp: the current point time_stamp\n
    @ imuLast/imuCur: the interval of [last,Cur] where the last point is within the interval\n
    @ return an tuple(last, Cur)\n
    """
    last = 0
    while imu_time_stamp[last] < point_time_stamp:
        last += 1
    
    return last, last+1


def numpy_proj(points, H_input = 64, W_input = 1800):
    degree2radian = math.pi / 180
    nLines = H_input
    AzimuthResolution = 360.0 / W_input # degree
    VerticalViewDown = -24.8
    VerticalViewUp = 2.0

    # specifications of Velodyne-64
    AzimuthResolution = AzimuthResolution * degree2radian # the original resolution is 0.18
    VerticalViewDown = VerticalViewDown * degree2radian
    VerticalViewUp = VerticalViewUp * degree2radian
    VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
    VerticalPixelsOffset = -VerticalViewDown / VerticalResolution

    # parameters for spherical ring's bounds
    PI = (np.pi)
    AzimuthResolution = (AzimuthResolution)
    VerticalPixelsOffset = (VerticalPixelsOffset)
    VerticalResolution = (VerticalResolution)

    r_f1 = np.linalg.norm(points[:, :2], ord=2, axis=1, keepdims = True)
    #cur_points = np.where( r_f1 > 35 , np.zeros_like(points), points)
    points_proj = points
    x = points_proj[:, 0]
    y = points_proj[:, 1]
    z = points_proj[:, 2]
    r = np.linalg.norm(points_proj, ord=2, axis=1)
    PC_project = np.zeros([H_input, W_input, 3]) # shape H W 3

    
    iCol = ((PI - np.arctan2(y,x)) / AzimuthResolution).astype(int) # alpha # 为什么要用Π减?
    beta = np.arcsin(z/r) # beta
    tmp_int = (beta / VerticalResolution + VerticalPixelsOffset).astype(int)
    iRow = H_input - tmp_int
    iRow = np.clip(iRow, 0, H_input - 1)
    iCol = np.clip(iCol, 0, W_input - 1)

    PC_project[iRow, iCol, :3] = points_proj
    return PC_project

def handlePointsData(points_list, imu_data, imu_time_stamp, timeStartStamps, timeEndStamps):
    """
    get the scanID and relTime of each point in one stamp\n
    assign points by different scanIDs\n
    remove distortion\n
    @ points_list: all stamp points Stamps x N x 4 the 3rd entry is reflectance data\n
    @ imu_data: S x 30 imu data\n
    @ imu_time_stamp: to get the last and current pointers of imu stamp\n
    @ timeStartStamps, timeEndStamps: to get the scanPeriod of this stamp\n
    @ return Stamps x N x 4 ndarray\n
    # HANDLE ALL STAMPS POINTS!!\n
    """
    
    points_len = len(points_list)
    points_all = np.zeros((points_len, 64*1800, 3))
    points_all_proj = np.zeros((points_len, 64, 1800, 3))
    for k in range(points_len):
        
        print("Begin to handle {:010}.txt points list----------------------".format(k))
        points = points_list[k]
        
        # handle every stamp
        cloudSize = (points.shape)[0]
        scanPeriod = timeEndStamps[k] - timeStartStamps[k]
        timeStamps_scan = timeStartStamps[k] + 0.5*scanPeriod
        timeStamps_start = timeStartStamps[k]
        imuPointerScanLast, imuPointerScanCur = getLastAndCur(imu_time_stamp, timeStamps_start)
        
        # Find the data at the beginning of each scan
        ratio_front_scan = (timeStamps_start - imu_time_stamp[imuPointerScanLast]) / (
                    imu_time_stamp[imuPointerScanCur] - imu_time_stamp[imuPointerScanLast])
            
        
        ratio_back_scan = 1.0 - ratio_front_scan

        imu_shift_x_start = ratio_front_scan * imu_data[imuPointerScanLast, 27] + ratio_back_scan * imu_data[imuPointerScanCur, 27]
        imu_shift_y_start = ratio_front_scan * imu_data[imuPointerScanLast, 28] + ratio_back_scan * imu_data[imuPointerScanCur, 28]
        imu_shift_z_start = ratio_front_scan * imu_data[imuPointerScanLast, 29] + ratio_back_scan * imu_data[imuPointerScanCur, 29]

        # the - sign means the clockwise spin and y/x is the counterclockwise spin
        start_original = -math.atan2(points[0, 1], points[0, 0])
        end_original = -math.atan2(points[-1, 1], points[-1, 0]) + 2 * np.pi
        
        if end_original - start_original > 3 * np.pi:
            end_original -= 2 * np.pi
        elif end_original - start_original < np.pi:
            end_original += 2 * np.pi


        points_proj = numpy_proj(points[:,:3])
        points_new = np.zeros((64*1800, 3))

        for i in range(1800):
            ori = np.pi*i*0.2/180

            relTime = (ori - start_original) / (end_original - start_original)

            point_time = relTime * scanPeriod
            
            point_time_stamp = timeStartStamps[k] + point_time - 0.5*scanPeriod

            imuPointerLast, imuPointerCur = getLastAndCur(imu_time_stamp, point_time_stamp)

            ratio_front = (point_time_stamp - imu_time_stamp[imuPointerLast]) / (
                    imu_time_stamp[imuPointerCur] - imu_time_stamp[imuPointerLast])
            
            ratio_back = 1.0 - ratio_front

            imu_shift_x_cur = ratio_front * imu_data[imuPointerLast, 27] + ratio_back * imu_data[imuPointerCur, 27]
            imu_shift_y_cur = ratio_front * imu_data[imuPointerLast, 28] + ratio_back * imu_data[imuPointerCur, 28]
            imu_shift_z_cur = ratio_front * imu_data[imuPointerLast, 29] + ratio_back * imu_data[imuPointerCur, 29]
            
            # calculate every point's distortion relative to the first point
            delta_shift_x_between_measure_and_cal = imu_shift_x_cur - imu_shift_x_start
            delta_shift_y_between_measure_and_cal = imu_shift_y_cur - imu_shift_y_start
            delta_shift_z_between_measure_and_cal = imu_shift_z_cur - imu_shift_z_start
            #print(delta_shift_between_measure_and_cal)
            delta_shift_between_measure_and_cal = np.array([delta_shift_z_between_measure_and_cal,
                                                            delta_shift_x_between_measure_and_cal,
                                                            delta_shift_y_between_measure_and_cal])
            
            delta_shift_between_measure_and_cal_list = np.tile(delta_shift_between_measure_and_cal, (64,1))

            points_proj[:, i, :] += delta_shift_between_measure_and_cal_list


        points_new = points_proj.reshape(64*1800, 3)
        np.savetxt(os.path.join(SAVE_DIR, '{:010d}_test_m.txt'.format(k)),points_new)
        points_all[k] = points_new
        points_all_proj[k] = points_proj

    return points_all, points_all_proj


if __name__ == '__main__':

    timeStartStamps = getTimeList(os.path.join(TEST_DIR, 'timestamps_start.txt'))
    timeEndStamps = getTimeList(os.path.join(TEST_DIR, 'timestamps_end.txt'))
    imu_time_stamp = getTimeList(os.path.join(IMU_DIR, 'timestamps.txt'))
    
    imu_data_not_handled = getImuData(IMU_FILES)
    imu_data = imuHandle(imu_data_not_handled, imu_time_stamp)


    points_list = [getPoints(test_file) for test_file in TEST_FILES]
    
    points_all, points_all_proj = handlePointsData(points_list, imu_data, imu_time_stamp, timeStartStamps, timeEndStamps)