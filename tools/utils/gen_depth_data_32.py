#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a script to generate depth data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import struct
try:
    from utils import *
except:
    from utils import *
import csv
import shutil
velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8



def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        # xyz = xyz @ R
        xyz = np.matmul(xyz, R)

    return xyz, xyzil['i']

def get_velo(velofile):
    return data2xyzi(np.fromfile(velofile))


def read_nuscenes_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('fffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)



def gen_depth_data(scan_folder, dst_folder, normalize=False):
    """ Generate projected range data in the shape of (64, 900, 1).
        The input raw data are in the shape of (Num_points, 3).
    """
    # specify the goal folder

    # load LiDAR scan files
    scan_paths = load_files(scan_folder)
    scan_files = os.listdir(scan_folder)

    depths = []
    
    # iterate over all scan files
    for idx in range(0, len(scan_paths)):
        
        # load a point cloud
        file_name = os.path.basename(scan_paths[idx])  # 获取文件名，包括扩展名
        file_name = os.path.splitext(file_name)[0]
        file_name = os.path.splitext(file_name)[0].split('/')[-1]
        # print(file_name)
        database_csv_path = "../../data/nuscenes/seq/00/pointcloud_locations.csv"
        query_csv_path = "../../data/nuscenes/seq/01/pointcloud_locations.csv"
        database_found = False
        query_found = False
        # with open(database_csv_path, 'r', newline='') as csvfile:
        #     reader = csv.DictReader(csvfile)
        #     for row in reader:
        #         timestamp = row['timestamp']
        #         if timestamp == file_name:
        #             database_found = True
        #             break
        with open(query_csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp = row['timestamp']
                if timestamp == file_name:
                    query_found = True
                    break

        current_vertex=read_nuscenes_velodyne(scan_paths[idx])
        fov_up = 10.67
        fov_down = -30.67
        proj_H = 32
        proj_W = 896
        lowest = 0.1
        highest = 6

        if False:
            current_vertex = get_velo(scan_paths[idx])[0]
            fov_up = 30.67
            fov_down = -10.67
            proj_H = 32
            proj_W = 896
            lowest = 0.1
            highest = 6
        start_time = time.time()
        proj_range, proj_vertex, _ = range_projection_HDL32E(current_vertex,
                                                      fov_up=fov_up,
                                                      fov_down=fov_down,
                                                      proj_H=proj_H,
                                                      proj_W=proj_W,
                                                      max_range=100,
                                                      cut_z=False,
                                                      low=lowest,
                                                      high=highest)

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        print("投影时间为:", execution_time_ms, "毫秒")

        # generate the destination path
        bin_filename = os.path.splitext(os.path.basename(scan_paths[idx]))[0]
        # if database_found == True:
        #     database_folder = os.path.join(dst_folder,str(0).zfill(2), 'depth_map')
        #     try:
        #         os.stat(database_folder)
        #     except:
        #         os.mkdir(database_folder)
        #     dst_path = os.path.join(database_folder, bin_filename) 
        #     # proj_range = cv2.resize(proj_range, (896, 64), interpolation=cv2.INTER_LINEAR)
        #     filename = dst_path + ".png"
        #     # cv2.imwrite(filename, proj_range)

        #     filename_bin = scan_folder + file_name + '.pcd.bin'
        #     bin_path = database_folder.replace('depth_map','PNV')
        #     shutil.copy(filename_bin, bin_path)
        #     print('finished generating database at: ', dst_path)
        if query_found == True:
            query_folder = os.path.join(dst_folder,str(1).zfill(2), 'depth_map')
            try:
                os.stat(query_folder)
            except:
                os.mkdir(query_folder)
            dst_path = os.path.join(query_folder, bin_filename) 
            # proj_range = cv2.resize(proj_range, (896, 64), interpolation=cv2.INTER_LINEAR)
            filename = dst_path + ".png"
            # cv2.imwrite(filename, proj_range)

            filename_bin = scan_folder + file_name + '.pcd.bin'
            bin_path = query_folder.replace('depth_map','PNV')
            shutil.copy(filename_bin, bin_path)
            print('finished generating query at: ', dst_path)
        else:  
            print('not available')
    return depths


if __name__ == '__main__':
    for idx in range(1, 2):
        # scan_folder ='/media/liji/T7/Datasets/NCLT/' + str(idx) + '/velodyne/' #'path_to_source_bin'
        # dst_folder ='/media/liji/T7/Datasets/NCLT/' + str(idx) #'path_to_saved_png'
        scan_folder ='../../data/nuscenes/samples/LIDAR_TOP/' #'path_to_source_bin'
        dst_folder ='../../data/nuscenes/seq/' #'path_to_saved_png'

        depth_data = gen_depth_data(scan_folder, dst_folder, normalize=False)

    
