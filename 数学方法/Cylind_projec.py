import numpy as np
import math
import matplotlib.pyplot as plt

def numpy_proj(PC, H_input = 64, W_input = 1800):
    degree2radian = math.pi / 180
    nLines = H_input
    AzimuthResolution = 360.0 / W_input # degree

    # kitti: -24.8-2.0 64lines
    # nuScenes: -30 - 10 32lines
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

    r_f1 = np.linalg.norm(PC[:, :2], ord=2, axis=1, keepdims = True)
    #cur_PC = np.where( r_f1 > 35 , np.zeros_like(PC), PC)
    cur_PC = PC
    x = cur_PC[:, 0]
    y = cur_PC[:, 1]
    z = cur_PC[:, 2]
    r = np.linalg.norm(cur_PC, ord=2, axis=1)
    PC_project = np.zeros([H_input, W_input, 3]) # shape H W 3

    
    iCol = ((PI - np.arctan2(y,x)) / AzimuthResolution).astype(np.int) # alpha # 为什么要用Π减?
    beta = np.arcsin(z/r) # beta
    tmp_int = (beta / VerticalResolution + VerticalPixelsOffset).astype(np.int)
    iRow = H_input - tmp_int
    iRow = np.clip(iRow, 0, H_input - 1)
    iCol = np.clip(iCol, 0, W_input - 1)
    print(iRow.max(),iRow.min())
    print(iCol.max(),iCol.min())
    PC_project[iRow, iCol, :3] = cur_PC
    return PC_project


if __name__ == '__main__':
    PC = np.loadtxt("D:/dataset/kitti_raw_data/unrectified/2011_09_30_drive_0016_extract/velodyne_points/data_modified_server/0000000001_m.txt",dtype=np.float32)[:, :3]
    PC_project = numpy_proj(PC, H_input = 64, W_input = 1800)
    norm2 = np.linalg.norm(PC_project[:,:,:3], ord=2, axis=2, keepdims = False)
    print(norm2.shape)
    plt.figure(figsize=(20,10))
    plt.imshow(norm2[:, :801])
    plt.xlim(0,800)
    plt.ylim(0,64)
    plt.show()
    print(PC_project.shape)