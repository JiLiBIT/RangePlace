# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse

def format_timestamp(value):
    return f"{value:06d}"

def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
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
        # df_locations['timestamp'] = df_locations['timestamp'].apply(format_timestamp)
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if folder in query_frames:
                if index in query_frames[folder]:
                    df_test = df_test._append(row, ignore_index=True)
            if folder in db_frames:
                if index in db_frames[folder]:
                    df_database = df_database._append(row, ignore_index=True)
        if df_database.shape[0] > 0:
            print(f"db:{df_database.shape[0]}")
            database_tree = KDTree(df_database[['northing', 'easting']])
            database_trees.append(database_tree)
        if df_test.shape[0] > 0:
            print(f"query:{df_test.shape[0]}")
            test_tree = KDTree(df_test[['northing', 'easting']])
            test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path + runs_folder + folder + "/" + filename), sep=',')
        # df_locations['timestamp'] = df_locations['timestamp'].apply(format_timestamp)
        df_locations['timestamp'] = runs_folder + folder + rangeimage_fols + \
                                    df_locations['timestamp'].astype(str) + '.png'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if folder in query_frames:
                if index in query_frames[folder]:
                    test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            if folder in db_frames:
                if index in db_frames[folder]:
                    database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                                'easting': row['easting']}
        
        
        if len(database.keys()) > 0:
            print("111",len(database.keys()))
            database_sets.append(database)
        if len(test.keys()) > 0:
            print("222",len(test.keys()))
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
                print("test_sets[j][key][i] ",test_sets[j][key][i] )
                # print(f"database:{i},test:{j},key:{key}")

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
    runs_folder = "/nclt/"
    all_folders = sorted(os.listdir(os.path.join(base_path + runs_folder)))
    index_list = [0,1]
    # index_list = [5]
    
    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])
    # db_frames = {'00': range(0,3000) ,'02': range(0,3400) '05': range(0,1000), '06': range(0,600)}
    # query_frames = {'00': range(3200, 4541),'02': range(3600,4661)'05': range(1200,2761), '06': range(800,1101)}

    db_frames = {'00': range(0,28128)}
    # query_frames = {'01': range(0, 28240),'02': range(0,16546), '03': range(0,24023)}
    query_frames = {'01': range(0, 28240)}
    
    print(folders)
    construct_query_and_database_sets(base_path, runs_folder, folders,  "/depth_map/",
                                      "pointcloud_locations.csv", db_frames, query_frames,"nclt")
