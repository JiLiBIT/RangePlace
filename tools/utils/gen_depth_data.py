#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of the project OverlapNet:
#https://github.com/PRBonn/OverlapNet
# Brief: a script to generate depth data
import os
from utils import load_files
import numpy as np
from utils import range_projection
import cv2
try:
    from utils import *
except:
    from utils import *

import scipy.linalg as linalg
def rotate_mat( axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix
    # print(type(rot_matrix))



def gen_depth_data(scan_folder, dst_folder, normalize=False):
    """ Generate projected range data in the shape of (64, 900, 1).
      The input raw data are in the shape of (Num_points, 3).
  """
    # specify the goal folder
    dst_folder = os.path.join(dst_folder, 'depth_map')
    try:
        os.stat(dst_folder)
        print('generating depth data in: ', dst_folder)
    except:
        print('creating new depth folder: ', dst_folder)
        os.mkdir(dst_folder)

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)
    # print(scan_paths)
    depths = []
    axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
    # iterate over all scan files
    for idx in range(len(scan_paths)):
        # load a point cloud


        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float64)
        current_vertex = np.float32(current_vertex)
        current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
        print(current_vertex.shape)
        current_vertex = current_vertex.reshape((-1, 3))
        print(current_vertex[1000])
        proj_range, _, _ = range_projection(current_vertex)  # proj_ranges   from larger to smaller

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)

        # generate the destination path
        bin_filename = os.path.splitext(os.path.basename(scan_paths[idx]))[0]
        dst_path = os.path.join(dst_folder, bin_filename) 
        # dst_path = os.path.join(dst_folder, str(idx).zfill(6))

        # np.save(dst_path, proj_range)
        filename = dst_path + ".png"
        cv2.imwrite(filename, proj_range)
        print('finished generating depth data at: ', dst_path)

    return depths


if __name__ == '__main__':
    for idx in range(2, 3):
        scan_folder ='/media/liji/T7/Datasets/NCLT/11/velodyne/' #'path_to_source_bin'
        dst_folder ='/media/liji/T7/Datasets/NCLT/11' #'path_to_saved_png'
        depth_data = gen_depth_data(scan_folder, dst_folder, normalize=False)

    
