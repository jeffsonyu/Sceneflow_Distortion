import numpy as np
import os
import open3d as o3d
from chamfer_loss import chamfer_distance_numpy, array2samples_distance
import random

def chamfer_loss(pc1, pc2, n):
    dists = np.zeros(n)

    source = o3d.geometry.PointCloud()

    pc1 = pc1[:, :3]
    pc2 = pc2[:, :3]
    len_1 = pc1.shape[0]
    len_2 = pc2.shape[0]

    for epoch in range(n):

        pc1_ = random.choices(pc1, k=106496)
        pc2_ = random.choices(pc2, k=106496)
        pc1_np = np.array(pc1_)
        pc2_np = np.array(pc2_)

        # 通过numpy读取txt点云
        source.points = o3d.utility.Vector3dVector(pc1_np)

        pc1_new = o3d.geometry.PointCloud.uniform_down_sample(source, 13)

        source.points = o3d.utility.Vector3dVector(pc2_np)
        pc2_new = o3d.geometry.PointCloud.uniform_down_sample(source, 13)

        pc1_new_np = np.asarray(pc1_new.points)
        pc2_new_np = np.asarray(pc2_new.points)


        dist = chamfer_distance_numpy(pc1_new_np, pc2_new_np)
        dists[epoch] = dist

    return dists, np.mean(dists)


if __name__ == "__main__":
    DIR_NAME = "E:/kitti_raw_data/unrectified/2011_10_03_drive_0034_extract/velodyne_points"
    DIR_MATH = os.path.join(DIR_NAME, "data_modified_math")
    DIR_SE = os.path.join(DIR_NAME, "data_sequence_txt")

    POINT_LIST_MATH = []
    for i in range(4661):
        file = "{:010}_test_m.txt".format(i+3)
        POINT_LIST_MATH.append(file)

    POINT_LIST = []
    for i in range(4661):
        file = "{:006}_r.txt".format(i)
        POINT_LIST.append(file)


    mean_list = []
    for i in range(1):
        pc1 = np.loadtxt(os.path.join(DIR_SE, POINT_LIST[3082]))
        pc2 = np.loadtxt(os.path.join(DIR_MATH, POINT_LIST_MATH[3085]))
        dists, mean = chamfer_loss(pc1, pc2, 10)
        mean_list.append(mean)

    print(mean_list)
    print(max(mean_list), mean_list.index(max(mean_list)))
    print(min(mean_list), mean_list.index(min(mean_list)))