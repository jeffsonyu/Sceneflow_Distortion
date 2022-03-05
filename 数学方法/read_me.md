## 数学方法点云去畸变

### 常量说明

```python
NSANCS = 64 # 雷达的线数
scnaPeriod = 0.1 # 雷达取点周期
VELO_TO_CAMERA # 将雷达坐标系转换为相机坐标系的T
```

### 函数说明

```python
R_from_RPY(roll, pitch, yaw) # 通过roll，pitch，yaw的顺序构造旋转矩阵R
R_from_YPR(yaw, pitch, roll) # 通过yaw，pitch，roll的顺序构造旋转矩阵
getTimeList(timeStamps) # 获取所有时间戳
getImuData(imu_files) # 获取IMU的信息，包括加速度，角速度
getPoints(test_file) # 从.txt文件中读取点云
getPointsFromBin(test_file) # 从.bin文件中读取点云
imuHandle(imu_data, imu_time_stamp) # 通过imu的时间戳以及imu的加速度信息得到imu从起点开始的位移，速度信息[IMU坐标系下]

getLastAndCur(imu_time_stamp, point_time_stamp, imuLast, imuCur) # 计算出离point_time_stamp最近的两个imu时间戳并返回下标

handlePointsData(points_list, imu_data, imu_time_stamp, timeStartStamps, timeEndStamps) # 通过imu的速度，位移，加速度等信息，转换到激光雷达坐标系下进行去畸变。大体的去畸变过程就是通过匀速运动对点云进行修正。具体实现可参照代码内部的注释
```

### 基本原理

激光雷达所采集得到的原始点云，点云的坐标是位于扫描结束时候的坐标系下的，在`cloudcompare`软件中能看到完整的一圈一圈的圆形。但是实际点云一定是一半在前一半在后，具体形状可以参考`KITTI`去除畸变的点云形状。因此使用匀速运动或者匀加速运动模型将那些滞后或者超前的点准确地校准，考虑到速度的精确性以及IMU的采样频率远高于激光雷达，IMU得到的速度以及加速度较为准确，通过相隔最近的两个IMU信息进行校准即使代码的主要想法。