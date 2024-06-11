# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from itertools import chain

def format_timestamp(value):
    return f"{value:06d}"

def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    # print(output)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, rangeimage_fols, filename, db_frames, query_frames, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path + runs_folder + folder + "/" + filename), sep=',')
        df_locations['timestamp'] = df_locations['timestamp']
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if index in query_frames[folder]:
                df_test = df_test.append(row, ignore_index=True)
            if index in db_frames[folder]:
                df_database = df_database.append(row, ignore_index=True)
        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path + runs_folder + folder + "/" + filename), sep=',')
        df_locations['timestamp'] = df_locations['timestamp']
        df_locations['timestamp'] = runs_folder + folder + rangeimage_fols + \
                                    df_locations['timestamp'].astype(str) + '.png'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if index in query_frames[folder]:
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            if index in db_frames[folder]:
                database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            # if i == j:
            #     continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=5)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()
                if len(test_sets[j][key][i] ) > 0:
                    print("key", key , "test_sets[j][key][i] ",test_sets[j][key][i] )
                # print("key",key)

    output_to_file(database_sets, base_path, "Hioformer_" + output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path, "Hioformer_" + output_name + '_evaluation_query.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    # For Oxford
    folders = []
    runs_folder = "/ford/"
    all_folders = sorted(os.listdir(os.path.join(base_path + runs_folder)))
    # index_list = [0,5,6]
    index_list = [1]
    
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])
    # db_frames = {'00': range(0,3000) ,'02': range(0,3400) '05': range(0,1000), '06': range(0,600)}
    # query_frames = {'00': range(3200, 4541),'02': range(3600,4661)'05': range(1200,2761), '06': range(800,1101)}

    # db_frames = {'01': range(450,2450)}
    # query_frames = {'01': list(chain(range(75, 400),range(2500, 2672)))}
    # db_frames = {'01': range(75,1500)}
    # query_frames = {'01': range(1700, 2672)}
    db_frames = {'02': range(1,2800)}
    query_frames = {'02': range(3000, 6103)}


    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders,  "/depth_map/",
                                      "pointcloud_locations.csv", db_frames, query_frames,"ford")
