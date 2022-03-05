import numpy as np
import os

if not os.path.exists('E:/kitti_raw_data/unrectified/2011_10_03_drive_0034_extract/velodyne_points/data_sequence_txt'):
    os.mkdir('E:/kitti_raw_data/unrectified/2011_10_03_drive_0034_extract/velodyne_points/data_sequence_txt')
for i in range(4661):
    file = '{:006d}.bin'.format(i)
    point = np.fromfile(os.path.join('E:/kitti_raw_data/sequence/02/velodyne', file), dtype=np.float32).reshape(-1, 4)
    np.savetxt(os.path.join('E:/kitti_raw_data/unrectified/2011_10_03_drive_0034_extract/velodyne_points/data_sequence_txt', '{:006d}_r.txt'.format(i)), point)
