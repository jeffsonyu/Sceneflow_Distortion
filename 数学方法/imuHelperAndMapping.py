import os
import numpy as np
import sys
from numpy import sin,cos,arctan, tan
import math
import datetime
from datetime import datetime
from concurrent import futures

# to get the undistorted points data in first point coordinate
# and save all points to files so that mapping can use it.
NSCANS = 64
scanPeriod = 0.1# velodyne scanperiod\appro 0.1s

POINT_LIST = []
for i in range(3):
    file = '{:010d}.txt'.format(i)
    POINT_LIST.append(file)

IMU_LIST = []
for i in range(2967):
    file = '{:010d}.txt'.format(i)
    IMU_LIST.append(file)

DIR_NAME = os.path.join("D:", "dataset", "kitti_raw_data", "unrectified")
DATA = os.path.join(DIR_NAME,"2011_09_30_drive_0016_extract")

TEST_DIR = os.path.join(DATA, "velodyne_points")
TEST_FILES = [file for file in POINT_LIST]

IMU_DIR = os.path.join(DATA, 'oxts')
IMU_FILES = [file for file in IMU_LIST]

if not os.path.exists(os.path.join(TEST_DIR, "data_modified_math")): os.mkdir(os.path.join(TEST_DIR, "data_modified_math"))
SAVE_DIR = os.path.join(TEST_DIR, "data_modified_math")
SAVE_DIR = os.path.join(TEST_DIR, "data_modified_math_test")

'''
gt_pose = np.loadtxt('04.txt').reshape(-1,3,4)
m,_,_ = gt_pose.shape
normalize = np.tile([0,0,0,1],[m,1]).reshape(-1,1,4)
# gt_pose S x 4 x 4 S is the time stamp
gt_pose = np.concatenate((gt_pose,normalize),axis=1) # camera coordinate T
'''

VELO_TO_CAMERA_0926 = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
            [1.480249e-02, 7.280733e-04,-9.998902e-01, -7.631618e-02],
            [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01], 
            [0,0,0,1.0000]])
VELO_TO_CAMERA_0930 = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03],
            [-2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02],
            [9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01], 
            [0,0,0,1.0000]])
VELO_TO_CAMERA_1003 = np.array([[7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02],
            [-2.771053e-03, 8.241710e-04, -9.999958e-01, -5.542117e-02],
            [9.999644e-01, 7.969825e-03, -2.764397e-03, -2.918589e-01], 
            [0,0,0,1.0000]])

VELO_TO_CAMERA = VELO_TO_CAMERA_0930

def R_from_RPY(roll, pitch, yaw):
    R_roll = np.array([[cos(roll), -sin(roll), 0],
                       [sin(roll),  cos(roll), 0],
                       [0,          0,         1]])

    R_pitch = np.array([[1,          0,         0],
                        [0, cos(pitch),sin(pitch)],
                        [0,-sin(pitch),cos(pitch)]])

    R_yaw = np.array([[cos(yaw), 0, -sin(yaw)],
                      [0,        1,         0],
                      [sin(yaw), 0,  cos(yaw)]])
    return R_yaw @ R_pitch @ R_roll


def R_from_YPR(yaw, pitch, roll):
    R_roll = np.array([[cos(roll), -sin(roll), 0],
                       [sin(roll),  cos(roll), 0],
                       [0,          0,         1]])

    R_pitch = np.array([[1,          0,         0],
                        [0, cos(pitch),sin(pitch)],
                        [0,-sin(pitch),cos(pitch)]])

    R_yaw = np.array([[cos(yaw), 0, -sin(yaw)],
                      [0,        1,         0],
                      [sin(yaw), 0,  cos(yaw)]])
    return R_yaw @ R_pitch @ R_roll


def getTimeList(timeStamps):
    with open(timeStamps,'r') as file:
        time_list = file.readlines()
        time_list_ = [datetime.timestamp(datetime.strptime(time[:-4],'%Y-%m-%d %H:%M:%S.%f')) for time in time_list]
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
    return np.asarray([np.loadtxt(os.path.join(IMU_DIR,'data',imu_file),dtype=np.float32) for imu_file in imu_files])


def getPoints(test_file):
    """
    get one stamp points  \n
    @return N x 4 ndarray
    """
    print("Handle {} data--------------------".format(test_file))
    return np.loadtxt(test_file,dtype=np.float32)


def getPointsFromBin(test_file):
    print("Handle {} data--------------------".format(test_file))
    return np.fromfile(test_file,dtype = np.float32).reshape(-1,4)
    

def imuHandle(imu_data, imu_time_stamp):
    """
    @ imu_data: imu imformation ndarray len(Stamps) x 30\n
    @ imu_time_stamp: imu_time stamp List[datetime.datetime]\n
    @ return: len(stamps) x 30 imu_data
    remove the gravity acceleration impact and calculate velocity and shift since first imu_stamp\n
    First remove g = 9.81m/s^2\n
    And then  calculate shift and velocity from 0 point\n
    a little change to oxtdata: imudata[24:27] are imushitX/Y/Z info and imu[27:30] are imuvelocityX/Y/Z
    """
    size = imu_data.shape[0] ## N -> the number of stamps
    for i in range(size):

        roll,pitch,yaw = imu_data[i,3],imu_data[i,4],imu_data[i,5]
        acceleration_x,acceleration_y,acceleration_z = imu_data[i,11],imu_data[i,12],imu_data[i,13]
        # remove the gravity
        # rotate to z-forward x-left y-up
        # because RPY is such coordinate
        acc_x = acceleration_y - np.sin(roll) * np.cos(pitch) * 9.81
        acc_y = acceleration_z - np.cos(roll) * np.cos(pitch) * 9.81
        acc_z = acceleration_x + np.sin(pitch) * 9.81
        imu_data[i,11],imu_data[i,12],imu_data[i,13] = acc_x,acc_y,acc_z

        
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
            imu_data[i,27], imu_data[i,28],imu_data[i,29] = 0.0, 0.0, 0.0
            imu_data[i,8], imu_data[i,9], imu_data[i,10] = 0.0, 0.0, 0.0
        else:
            time_diff = imu_time_stamp[i] - imu_time_stamp[i-1]
            if time_diff < scanPeriod:
                imu_data[i,27] = imu_data[i-1,27] + imu_data[i-1,8] * time_diff + 0.5 * acc_x * time_diff * time_diff
                imu_data[i,28] = imu_data[i-1,28] + imu_data[i-1,9] * time_diff + 0.5 * acc_y * time_diff * time_diff
                imu_data[i,29] = imu_data[i-1,29] + imu_data[i-1,10] * time_diff + 0.5 * acc_z * time_diff * time_diff

                imu_data[i,8] = imu_data[i-1,8] + acc_x * time_diff
                imu_data[i,9] = imu_data[i-1,9] + acc_y * time_diff
                imu_data[i,10] = imu_data[i-1,10] + acc_y * time_diff
        print(imu_data[i,27:])
    return imu_data
    

def getLastAndCur(imu_time_stamp, point_time_stamp, imuLast, imuCur):
    """
    @ imu_time_stamp: the list of imu_time_stamp\n
    @ point_time_stamp: the current point time_stamp\n
    @ imuLast/imuCur: the interval of [last,Cur] where the last point is within the interval\n
    @ return an tuple(last, Cur)\n
    """
    Cur = imuCur
    last = 0
    while imu_time_stamp[Cur] < point_time_stamp:
        Cur += 1
    for i in range(Cur,imuLast,-1):
        if imu_time_stamp[i] < point_time_stamp:
            last = i
            break
    return last,Cur


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
    global imuPointerLast, imuPointerCur
    points_len = len(points_list)
    for k in range(points_len):
        print("Begin to handle {:010}.txt points list----------------------".format(k))
        points = points_list[k]
        # handle every stamp
        cloudSize = (points.shape)[0]
        scanPeriod = timeEndStamps[k] - timeStartStamps[k]
        # the - sign means the clockwise spin and y/x is the counterclockwise spin
        start_original = -math.atan2(points[0,1],points[0,0])
        end_original = -math.atan2(points[-1,1],points[-1,0]) + 2 * np.pi

        if end_original - start_original > 3*np.pi:
            end_original -= 2*np.pi
        elif end_original - start_original < np.pi:
            end_original += 2*np.pi

        halfPassed = False
        count = cloudSize
        validPoints = [[] for i in range(NSCANS)]

        for i in range(cloudSize):
            
            
            # to handle every point in current stamp
            # cloudSize is the number of points
            # point is the current point
            point = points[i]
            # print(point)
            point[0], point[1], point[2] = point[1],point[2],point[0]
            # print(point)
            angle = np.arctan(point[1] / np.sqrt(point[0]*point[0]+point[2]*point[2])) * 180 / np.pi

            roundedAngle = int(angle + 0.5) if angle > 0.0 else int(angle-0.5)
            # filter the points which are beyond the 64-line 
            if roundedAngle > 0:
                scanID = roundedAngle
            else:
                scanID = roundedAngle + NSCANS - 1

            if (scanID > NSCANS - 1) or scanID < 0:
                count -= 1
                continue

            ori = -math.atan2(point[0],point[2])
            if not halfPassed:
                # verify -pi/2< ori - start_original < 3*pi/2
                if (ori < start_original - np.pi/2):
                    ori += 2*np.pi
                elif ori > start_original + np.pi * 3 / 2:
                    ori -= 2*np.pi

                if ori - start_original > np.pi:
                    halfPassed = True
            else:
                ori += 2*np.pi
                # verify -3*pi/2 < ori - end_original < pi/2
                if ori < end_original - np.pi * 3 / 2:
                    ori += 2 * np.pi
                elif ori > end_original + np.pi / 2:
                    ori -= 2 * np.pi
            relTime = (ori - start_original) / (end_original-start_original)
            # point[3] represent the line index and relative time in this stamp
            point[3] = scanID + scanPeriod * relTime
            ## these steps are to remove the distortion

            
            point_time = relTime * scanPeriod
            point_time_stamp = timeStartStamps[k] + point_time - 0.5*scanPeriod
            imuPointerLast, imuPointerCur = getLastAndCur(imu_time_stamp,point_time_stamp,imuPointerLast,imuPointerCur)
            # begin to interpolate
            # two coefficent from imustart and imuend
            # all are in world coornidate z-front x-lefty-up
            ratio_front = (point_time_stamp - imu_time_stamp[imuPointerLast]) / (imu_time_stamp[imuPointerCur] - imu_time_stamp[imuPointerLast])
            # ratio_back = (imu_time_stamp[imuPointerCur] - point_time_stamp) / (imu_time_stamp[imuPointerCur] - imu_time_stamp[imuPointerLast])
            ratio_back = 1.0 - ratio_front 
            imu_roll_cur = ratio_front * imu_data[imuPointerLast,3] + ratio_back * imu_data[imuPointerCur,3]
            imu_pitch_cur = ratio_front * imu_data[imuPointerLast,4] + ratio_back * imu_data[imuPointerCur,4]
            imu_yaw_cur = ratio_front * imu_data[imuPointerLast,5] + ratio_back * imu_data[imuPointerCur,5]

            imu_shift_x_cur = ratio_front * imu_data[imuPointerLast,-3] + ratio_back * imu_data[imuPointerCur,-3]
            imu_shift_y_cur = ratio_front * imu_data[imuPointerLast,-2] + ratio_back * imu_data[imuPointerCur,-2]
            imu_shift_z_cur = ratio_front * imu_data[imuPointerLast,-1] + ratio_back * imu_data[imuPointerCur,-1]

            imu_velocity_x_cur = ratio_front * imu_data[imuPointerLast,8] + ratio_back * imu_data[imuPointerCur,8]
            imu_velocity_y_cur = ratio_front * imu_data[imuPointerLast,9] + ratio_back * imu_data[imuPointerCur,9]
            imu_velocity_z_cur = ratio_front * imu_data[imuPointerLast,10] + ratio_back * imu_data[imuPointerCur,10]

            if i == 0:
                # the first point can be used to get imuStart -- the first point coordinate
                imu_roll_start = imu_roll_cur
                imu_pitch_start = imu_pitch_cur
                imu_yaw_start = imu_yaw_cur

                imu_shift_x_start = imu_shift_x_cur
                imu_shift_y_start = imu_shift_y_cur
                imu_shift_z_start = imu_shift_z_cur

                imu_velocity_x_start = imu_velocity_x_cur
                imu_velocity_y_start = imu_velocity_y_cur
                imu_velocity_z_start = imu_velocity_z_cur

                point[0],point[1],point[2] = point[2],point[0],point[1]
                points[i] = point
            else:
                # calculate every point's distortion relative to the first point
                delta_shift_x_between_measure_and_cal = imu_shift_x_cur - imu_shift_x_start - imu_velocity_x_cur * point_time
                delta_shift_y_between_measure_and_cal = imu_shift_y_cur - imu_shift_y_start - imu_velocity_y_cur * point_time
                delta_shift_z_between_measure_and_cal = imu_shift_z_cur - imu_shift_z_start - imu_velocity_z_cur * point_time
                # rotate the delta shift from world coordinate to imu first coordinate
                # get the rotation matrix
                YPR = R_from_YPR(-imu_yaw_start,-imu_pitch_start,-imu_roll_start)
                RPY = R_from_RPY(imu_roll_cur,imu_pitch_cur,imu_yaw_cur)

                # rotate to imu first coordinate
                # x_world = YPR* x_first
                # x_first = R^-1*P^-1*Y^-1 * x_world
                delta_shift_between_measure_and_cal = np.array([delta_shift_x_between_measure_and_cal,
                delta_shift_y_between_measure_and_cal,
                delta_shift_z_between_measure_and_cal])
                #delta_shift_between_measure_and_cal = YPR @ delta_shift_between_measure_and_cal
                
                delta_velocity_x_between_measure_and_cal = imu_velocity_x_cur - imu_velocity_x_start - imu_velocity_x_cur * point_time
                delta_velocity_y_between_measure_and_cal = imu_velocity_y_cur - imu_velocity_y_start - imu_velocity_y_cur * point_time
                delta_velocity_z_between_measure_and_cal = imu_velocity_z_cur - imu_velocity_z_start - imu_velocity_z_cur * point_time
                # rotate the delta velocity from world coordinate  to imu first coordinate

                delta_velocity_between_measure_and_cal = np.array([delta_velocity_x_between_measure_and_cal,
                delta_velocity_y_between_measure_and_cal,
                delta_velocity_z_between_measure_and_cal])

                delta_velocity_between_measure_and_cal = YPR @ delta_velocity_between_measure_and_cal
                # remove point distortion
                # point in velodyne coordinate system
                # rotate to world coordinate and rotate to imu start coordinate
                # we can get every point x,y,z in the first point coordinate
                RPY = np.hstack((RPY,np.zeros((3,1))))
                YPR = np.hstack((YPR,np.zeros((3,1))))
                RPY = np.vstack((RPY,np.array([0,0,0,1])))
                YPR = np.vstack((YPR,np.array([0,0,0,1])))

                point = YPR @ point
                # now point is in imu first coordinate
                if halfPassed:
                    point[:3] += point_time * np.asarray([imu_velocity_x_start,
                                                          imu_velocity_y_start,
                                                          imu_velocity_z_start])
                else:
                    point[:3] -= point_time * np.asarray([imu_velocity_x_start,
                                                          imu_velocity_y_start,
                                                          imu_velocity_z_start])
                # now point has removed velodyne distortion
                #print(delta_shift_between_measure_and_cal)
                point[:3] += 50*delta_shift_between_measure_and_cal
                # now p just removed constant speed distortion
                # now p is in imu first coordinate
                # now rotate to global and rotate to velodyne
                R_to_velodyne = R_from_RPY(imu_roll_start,imu_pitch_start,imu_yaw_start)
                R_to_velodyne = np.hstack((R_to_velodyne,np.zeros((3,1))))
                R_to_velodyne = np.vstack((R_to_velodyne,np.array([0,0,0,1])))
                point = R_to_velodyne @ point
                point[0],point[1],point[2] = point[2],point[0],point[1]
                points[i] = point
                #print(point)
            # validPoints[scanID].append(point)
        #points_list[k] = (VELO_TO_CAMERA @ points_list[k].T).T
        
        np.savetxt(os.path.join(SAVE_DIR, '{:010d}_test_m.txt'.format(k)),points_list[k])
    return np.asarray(points_list)



if __name__ == '__main__':
    # List[float] -- store timestamps
    # timeStamps = getTimeList(os.path.join(TEST_DIR,"timestamps.txt"))
    # timeStartStamps = getTimeList(os.path.join(TEST_DIR,"timestamps_start.txt"))
    # timeEndStamps = getTimeList(os.path.join(TEST_DIR,"timestamps_end.txt"))
    # # this is the imu data of each stamp
    # imu_time_stamp = getTimeList(IMU_DIR + '/timestamps.txt')
    # imu_data_not_handled = getImuData(IMU_FILES)
    # print("Begin to get imu data-------------------------------")
    # imu_data = imuHandle(imu_data_not_handled,imu_time_stamp)

    # points_len = len(TEST_FILES)
    # points_list = [getPoints(os.path.join(TEST_DIR,'data',TEST_FILES[i])) for i in range(2)]
    
    # imuPointerLast = 0
    # imuPointerCur = 0
    # points_all_stamp = handlePointsData(points_list,imu_data,imu_time_stamp,timeStartStamps,timeEndStamps)
    # for i in range(2):
    #     np.savetxt(os.path.join(TEST_DIR,'{:010}.txt'.format(i)),points_all_stamp[i])
    timeStartStamps = getTimeList(os.path.join(TEST_DIR,'timestamps_start.txt'))
    timeEndStamps = getTimeList(os.path.join(TEST_DIR,'timestamps_end.txt'))
    imu_time_stamp = getTimeList(os.path.join(IMU_DIR,'timestamps.txt'))
    
    imu_data_not_handled = getImuData(IMU_FILES)
    imu_data = imuHandle(imu_data_not_handled,imu_time_stamp)
    imuPointerCur = 0
    imuPointerLast = 0
    points_list = [getPoints(os.path.join(TEST_DIR,'data',test_file)) for test_file in TEST_FILES]
    # points_list = [np.fromfile('0000000000.bin',dtype=np.float32).reshape(-1,4),
    #                np.fromfile('0000000001.bin',dtype=np.float32).reshape(-1,4)]
    
    points_all_stamp = handlePointsData(points_list,imu_data,imu_time_stamp,timeStartStamps,timeEndStamps)
    # for i in range(2):
    #     np.tofile('{:010}_m.bin'.format(i))
    # print(points)

