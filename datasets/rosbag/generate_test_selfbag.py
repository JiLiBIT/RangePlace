import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def format_timestamp(value):
    return f"{value:06d}"


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(
    base_path,
    runs_folder,
    folders,
    rangeimage_fols,
    filename,
    db_frames,
    query_frames,
    output_name,
):
    # 初始化数据
    database_trees = []
    test_trees = []

    # 遍历文件夹
    for folder in folders:
        print(folder)

        # 构建 pd ,读取 csv
        df_database = pd.DataFrame(columns=["file", "northing", "easting"])
        df_test = pd.DataFrame(columns=["file", "northing", "easting"])

        df_locations = pd.read_csv(
            os.path.join(base_path + runs_folder + folder + "/" + filename), sep=","
        )
        df_locations["timestamp"] = df_locations["timestamp"].apply(format_timestamp)

        # 划分测试集和训练集
        for index, row in df_locations.iterrows():
            if index in db_frames[folder]:
                df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
            elif index in query_frames[folder]:
                df_database = pd.concat(
                    [df_database, row.to_frame().T], ignore_index=True
                )

        # 创建 KDTree
        database_tree = KDTree(df_database[["northing", "easting"]])
        test_tree = KDTree(df_test[["northing", "easting"]])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []

    # 遍历文件夹，读取深度图
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(
            os.path.join(base_path + runs_folder + folder + "/" + filename), sep=","
        )
        df_locations["timestamp"] = df_locations["timestamp"].apply(format_timestamp)
        df_locations["timestamp"] = (
            runs_folder
            + folder
            + rangeimage_fols
            + df_locations["timestamp"].astype(str)
            + ".png"
        )
        df_locations = df_locations.rename(columns={"timestamp": "file"})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if index in query_frames[folder]:
                test[len(test.keys())] = {
                    "query": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                }
            if index in db_frames[folder]:
                database[len(database.keys())] = {
                    "query": row["file"],
                    "northing": row["northing"],
                    "easting": row["easting"],
                }
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            # if i == j:
            #     continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array(
                    [[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]]
                )
                index = tree.query_radius(coor, r=1.5)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()
                print("test_sets[j][key][i] ", test_sets[j][key][i])
                # print("key",key)

    output_to_file(
        database_sets,
        base_path,
        "Hioformer_" + output_name + "_evaluation_database.pickle",
    )
    output_to_file(
        test_sets, base_path, "Hioformer_" + output_name + "_evaluation_query.pickle"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation datasets")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root folder"
    )

    args = parser.parse_args()
    print("Dataset root: {}".format(args.dataset_root))

    assert os.path.exists(
        args.dataset_root
    ), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    # For Kitti
    folders = []
    runs_folder = "/data_root_folder/"
    all_folders = sorted(os.listdir(os.path.join(base_path + runs_folder)))
    index_list = [1, 3]

    print(len(index_list))
    for index in index_list:
        folders.append(all_folders[index])

    db_frames = {
        "01": range(0, 100),
        "03": range(0, 100),
    }
    query_frames = {
        "01": range(101, 16346),
        "03": range(101, 2299),
    }

    print(folders)
    construct_query_and_database_sets(
        base_path,
        runs_folder,
        folders,
        "/depth_map/",
        "pointcloud_locations.csv",
        db_frames,
        query_frames,
        "kitti",
    )
