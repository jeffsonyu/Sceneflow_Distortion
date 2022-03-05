import os
import numpy as np
import sys
from numpy import float32, sin, cos, arctan, tan
import math
import datetime
from datetime import datetime
from concurrent import futures

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


DIR_NAME = "E:/dataset/kitti_raw_data/unrectified/"
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
            time_diff = imu_time_stamp[i] - imu_time_stamp[i - 1]
            if time_diff < scanPeriod:
                imu_data[i, 27] = imu_data[i - 1, 27] + imu_data[
                    i - 1, 8] * time_diff + 0.5 * acc_x * time_diff * time_diff
                imu_data[i, 28] = imu_data[i - 1, 28] + imu_data[
                    i - 1, 9] * time_diff + 0.5 * acc_y * time_diff * time_diff
                imu_data[i, 29] = imu_data[i - 1, 29] + imu_data[
                    i - 1, 10] * time_diff + 0.5 * acc_z * time_diff * time_diff

                imu_data[i, 8] = imu_data[i - 1, 8] + acc_x * time_diff
                imu_data[i, 9] = imu_data[i - 1, 9] + acc_y * time_diff
                imu_data[i, 10] = imu_data[i - 1, 10] + acc_z * time_diff
        #print(imu_data[i, 27:])
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

        imu_roll_start = ratio_front_scan * imu_data[imuPointerScanLast, 3] + ratio_back_scan * imu_data[imuPointerScanCur, 3]
        imu_pitch_start = ratio_front_scan * imu_data[imuPointerScanLast, 4] + ratio_back_scan * imu_data[imuPointerScanCur, 4]
        imu_yaw_start = ratio_front_scan * imu_data[imuPointerScanLast, 5] + ratio_back_scan * imu_data[imuPointerScanCur, 5]

        imu_shift_x_start = ratio_front_scan * imu_data[imuPointerScanLast, 27] + ratio_back_scan * imu_data[imuPointerScanCur, 27]
        imu_shift_y_start = ratio_front_scan * imu_data[imuPointerScanLast, 28] + ratio_back_scan * imu_data[imuPointerScanCur, 28]
        imu_shift_z_start = ratio_front_scan * imu_data[imuPointerScanLast, 29] + ratio_back_scan * imu_data[imuPointerScanCur, 29]
            
        imu_velocity_x_start = ratio_front_scan * imu_data[imuPointerScanLast, 8] + ratio_back_scan * imu_data[imuPointerScanCur, 8]
        imu_velocity_y_start = ratio_front_scan * imu_data[imuPointerScanLast, 9] + ratio_back_scan * imu_data[imuPointerScanCur, 9]
        imu_velocity_z_start = ratio_front_scan * imu_data[imuPointerScanLast, 10] + ratio_back_scan * imu_data[imuPointerScanCur, 10]

        # the - sign means the clockwise spin and y/x is the counterclockwise spin
        start_original = -math.atan2(points[0, 1], points[0, 0])
        end_original = -math.atan2(points[-1, 1], points[-1, 0]) + 2 * np.pi
        
        if end_original - start_original > 3 * np.pi:
            end_original -= 2 * np.pi
        elif end_original - start_original < np.pi:
            end_original += 2 * np.pi

        halfPassed = False
        count = cloudSize

        for i in range(cloudSize):

            # to handle every point in current stamp
            # cloudSize is the number of points
            # point is the current point
            
            point = points[i]
            
            point[0], point[1], point[2] = point[1], point[2], point[0]

            angle = np.arctan(point[1] / np.sqrt(point[0] * point[0] + point[2] * point[2])) * 180 / np.pi

            roundedAngle = int(angle + 0.5) if angle > 0.0 else int(angle - 0.5)
            # filter the points which are beyond the 64-line 
            if roundedAngle > 0:
                scanID = roundedAngle
            else:
                scanID = roundedAngle + NSCANS - 1

            if (scanID > NSCANS - 1) or scanID < 0:
                count -= 1
                continue

            ori = -math.atan2(point[0], point[2]) - np.pi
            #print(ori*180/np.pi)
            if ori <= 0:
                ori += 2 * np.pi
            elif ori >= 2*np.pi:
                ori -= 2 * np.pi
            relTime = (ori - start_original) / (end_original - start_original)
            #print(relTime)
            # point[3] represent the line index and relative time in this stamp
            point[3] = np.round(scanID + scanPeriod * relTime)
            
            ## these steps are to remove the distortion

            point_time = relTime * scanPeriod
            
            point_time_stamp = timeStartStamps[k] + point_time - 0.5*scanPeriod

            imuPointerLast, imuPointerCur = getLastAndCur(imu_time_stamp, point_time_stamp)
            # begin to interpolate
            # two coefficent from imustart and imuend
            # all are in world coornidate z-front x-left y-up
            ratio_front = (point_time_stamp - imu_time_stamp[imuPointerLast]) / (
                    imu_time_stamp[imuPointerCur] - imu_time_stamp[imuPointerLast])
            
            ratio_back = 1.0 - ratio_front

            imu_roll_cur = ratio_front * imu_data[imuPointerLast, 3] + ratio_back * imu_data[imuPointerCur, 3]
            imu_pitch_cur = ratio_front * imu_data[imuPointerLast, 4] + ratio_back * imu_data[imuPointerCur, 4]
            imu_yaw_cur = ratio_front * imu_data[imuPointerLast, 5] + ratio_back * imu_data[imuPointerCur, 5]

            imu_shift_x_cur = ratio_front * imu_data[imuPointerLast, 27] + ratio_back * imu_data[imuPointerCur, 27]
            imu_shift_y_cur = ratio_front * imu_data[imuPointerLast, 28] + ratio_back * imu_data[imuPointerCur, 28]
            imu_shift_z_cur = ratio_front * imu_data[imuPointerLast, 29] + ratio_back * imu_data[imuPointerCur, 29]
            
            imu_velocity_x_cur = ratio_front * imu_data[imuPointerLast, 8] + ratio_back * imu_data[imuPointerCur, 8]
            imu_velocity_y_cur = ratio_front * imu_data[imuPointerLast, 9] + ratio_back * imu_data[imuPointerCur, 9]
            imu_velocity_z_cur = ratio_front * imu_data[imuPointerLast, 10] + ratio_back * imu_data[imuPointerCur, 10]

            imu_shift_cur = np.array([imu_shift_x_cur, imu_shift_y_cur, imu_shift_z_cur])
            
            # calculate every point's distortion relative to the first point
            delta_shift_x_between_measure_and_cal = imu_shift_x_cur - imu_shift_x_start# - imu_velocity_x_start * point_time
            delta_shift_y_between_measure_and_cal = imu_shift_y_cur - imu_shift_y_start# - imu_velocity_y_start * point_time
            delta_shift_z_between_measure_and_cal = imu_shift_z_cur - imu_shift_z_start# - imu_velocity_z_start * point_time
            # rotate the delta shift from world coordinate to imu first coordinate
            # get the rotation matrix
            YPR = R_from_YPR(-imu_yaw_start, -imu_pitch_start, -imu_roll_start)
            RPY = R_from_RPY(imu_roll_cur, imu_pitch_cur, imu_yaw_cur)

            # rotate to imu first coordinate
            # x_world = YPR* x_first
            # x_first = R^-1*P^-1*Y^-1 * x_world
            delta_shift_between_measure_and_cal = np.array([delta_shift_x_between_measure_and_cal,
                                                            delta_shift_y_between_measure_and_cal,
                                                            delta_shift_z_between_measure_and_cal])
            
            
            #delta_shift_between_measure_and_cal = YPR @ delta_shift_between_measure_and_cal
            #print(delta_shift_between_measure_and_cal)
            delta_velocity_x_between_measure_and_cal = imu_velocity_x_cur - imu_velocity_x_start
            delta_velocity_y_between_measure_and_cal = imu_velocity_y_cur - imu_velocity_y_start
            delta_velocity_z_between_measure_and_cal = imu_velocity_z_cur - imu_velocity_z_start
            # rotate the delta velocity from world coordinate  to imu first coordinate

            delta_velocity_between_measure_and_cal = np.array([delta_velocity_x_between_measure_and_cal,
                                                               delta_velocity_y_between_measure_and_cal,
                                                               delta_velocity_z_between_measure_and_cal])
            
            delta_velocity_between_measure_and_cal = YPR @ delta_velocity_between_measure_and_cal
            
            # remove point distortion
            # point in velodyne coordinate system
            # rotate to world coordinate and rotate to imu start coordinate
            # we can get every point x,y,z in the first point coordinate
            RPY = np.hstack((RPY, np.zeros((3, 1))))
            YPR = np.hstack((YPR, np.zeros((3, 1))))
            RPY = np.vstack((RPY, np.array([0, 0, 0, 1])))
            YPR = np.vstack((YPR, np.array([0, 0, 0, 1])))

            #point = YPR @ point
                
            # now point is in imu first coordinate
                
            # if halfPassed:
            '''
            point[:3] += (point_time - 0.5*scanPeriod) * np.asarray([imu_velocity_x_start,
                                                          imu_velocity_y_start,
                                                          imu_velocity_z_start])
            '''

            # now point has removed velodyne distortion
            point[:3] += delta_shift_between_measure_and_cal
            # now p just removed constant speed distortion
            # now p is in imu first coordinate
            # now rotate to global and rotate to velodyne
            R_to_velodyne = R_from_RPY(imu_roll_cur, imu_pitch_cur, imu_yaw_cur)
            R_to_velodyne = np.hstack((R_to_velodyne, np.zeros((3, 1))))
            R_to_velodyne = np.vstack((R_to_velodyne, np.array([0, 0, 0, 1])))
            #point = R_to_velodyne @ point
            point[0], point[1], point[2] = point[2], point[0], point[1]
                
            points[i] = point

        np.savetxt(os.path.join(SAVE_DIR, '{:010d}_test_m.txt'.format(k)),points_list[k])
    return np.asarray(points_list)


if __name__ == '__main__':

    timeStartStamps = getTimeList(os.path.join(TEST_DIR, 'timestamps_start.txt'))
    timeEndStamps = getTimeList(os.path.join(TEST_DIR, 'timestamps_end.txt'))
    imu_time_stamp = getTimeList(os.path.join(IMU_DIR, 'timestamps.txt'))
    
    imu_data_not_handled = getImuData(IMU_FILES)
    imu_data = imuHandle(imu_data_not_handled, imu_time_stamp)


    points_list = [getPoints(test_file) for test_file in TEST_FILES]
    
    points_all_stamp = handlePointsData(points_list, imu_data, imu_time_stamp, timeStartStamps, timeEndStamps)
