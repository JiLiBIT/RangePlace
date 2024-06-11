# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm

from datasets.base_datasets import TrainingTuple
# Import test set boundaries
# from datasets.pointnetvlad.generate_test_kitti import P1, P2, P3, P4, check_in_test_set

# Test set boundaries
# P = [P1, P2, P3, P4]

RUNS_FOLDER = "/data_root_folder/"
FILENAME = "pointcloud_locations.csv"
RANGEIMAGE_FOLS = "/depth_map/"


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=5):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    # KITTI dataset parameters in the original PointNetVLAD code: ind_nn_r=5, ind_r=25
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.png', f"Expected .png file: {scan_filename}"
        timestamp = str(os.path.splitext(scan_filename)[0]).zfill(6)
        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def format_timestamp(value):
    return f"{value:06d}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    all_folders = sorted(os.listdir(os.path.join(base_path + RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders) - 1)
    # index_list = [0,2,5,6]
    print("Number of runs: " + str(len(index_list)))

    
    for index in index_list:
        folders.append(all_folders[index])


    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        
        df_locations = pd.read_csv(os.path.join(base_path + RUNS_FOLDER + folder + "/" + FILENAME), sep=',')
        df_locations['timestamp'] = df_locations['timestamp'].apply(format_timestamp)
        
        
        df_locations['timestamp'] = RUNS_FOLDER + folder + RANGEIMAGE_FOLS + df_locations['timestamp'].astype(str) + '.png'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        
        db_frames = {'00': range(0,3000), '01': range(0, 1101), '02': range(0,4661), '03': range(0,801),
                     '04': range(0,271), '05': range(0,1000), '06': range(0,600), '07': range(0,1101),
                     '08': range(0,4071), '09': range(0,1591),'10': range(0,1201)}
        query_frames = {'00': range(3200, 4541),'05': range(1200,2761), '06': range(800,1101)}

        # db_frames = {'00': range(0,4541), '01': range(0, 1101), '02': range(0,4661), '03': range(0,801),
        #              '04': range(0,271), '05': range(0,1100), '06': range(0,1101), '07': range(0,1101),
        #              '08': range(0,4071), '09': range(0,1591),'10': range(0,1201),}
        # query_frames = {'05': range(1200,2761)}


        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if index in db_frames[folder]:
                df_train = df_train._append(row, ignore_index=True)
            elif index in query_frames[folder]:
                df_test = df_test._append(row, ignore_index=True)
            
        
        # for index, row in df_locations.iterrows():
        #     if folder == "00" or folder == "05" or folder == "06":
        #         df_test = df_test.append(row, ignore_index=True)
        #     else:
        #         df_train = df_train.append(row, ignore_index=True)
        print(len(df_train['file']))
        print(len(df_test['file']))
    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_kitti.pickle", ind_nn_r=5)
    construct_query_dict(df_test, base_path, "test_queries_kitti.pickle", ind_nn_r=5)
