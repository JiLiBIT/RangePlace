import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm

from datasets.base_datasets import TrainingTuple

RUNS_FOLDER = "/data_root_folder/"
FILENAME = "pointcloud_locations.csv"
RANGEIMAGE_FOLS = "/depth_map/"


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=5):

    # ind_nn_r: 正样本阈值
    # ind_r_r: 负样本阈值
    # 原始 PointNetVLAD 代码中的基线数据集参数: ind_nn_r=10, ind_r=50
    # 原始 PointNetVLAD 代码中的精细数据集参数: ind_nn_r=12.5, ind_r=50
    # 原始 PointNetVLAD 代码中的 KITTI 数据集参数 : ind_nn_r=5, ind_r=25

    # 构建 KD 树:
    tree = KDTree(df_centroids[["northing", "easting"]])

    # 查询半径:查询正负样本的序列
    ind_nn = tree.query_radius(df_centroids[["northing", "easting"]], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[["northing", "easting"]], r=ind_r_r)

    # 初始化查询字典:
    queries = {}

    # 遍历每个正样本序列:
    for anchor_ndx in range(len(ind_nn)):
        # 获取正样本的位置
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][["northing", "easting"]])
        # 获取正样本的文件路径
        query = df_centroids.iloc[anchor_ndx]["file"]
        # 从文件中提取时间戳
        scan_filename = os.path.split(query)[1]

        # 确保文件名以 .png 结尾
        assert (
            os.path.splitext(scan_filename)[1] == ".png"
        ), f"Expected .png file: {scan_filename}"

        # 格式化时间戳
        timestamp = str(os.path.splitext(scan_filename)[0]).zfill(6)

        # 提取正样本索引
        positives = ind_nn[anchor_ndx]
        # 提取负样本索引
        non_negatives = ind_r[anchor_ndx]

        # 移除自身索引
        positives = positives[positives != anchor_ndx]

        # 对正负样本进行升序排序
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # 创建查询元组
        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            position=anchor_pos,
        )

    file_path = os.path.join(base_path, filename)
    with open(file_path, "wb") as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


# 将 value 转换为字符串
def format_timestamp(value):
    return f"{value:06d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Baseline training dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root folder"
    )
    args = parser.parse_args()
    print("Dataset root: {}".format(args.dataset_root))

    assert os.path.exists(
        args.dataset_root
    ), f"Cannot access dataset root folder: {args.dataset_root}"

    base_path = args.dataset_root
    all_folders = sorted(os.listdir(os.path.join(base_path + RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders) - 1)

    print("Number of runs: " + str(len(index_list)))

    for index in index_list:
        folders.append(all_folders[index])

    df_train = pd.DataFrame(columns=["file", "northing", "easting"])
    df_test = pd.DataFrame(columns=["file", "northing", "easting"])

    # 遍历文件夹
    for folder in tqdm.tqdm(folders):
        # 读取 CSV 文件
        df_locations = pd.read_csv(
            os.path.join(base_path + RUNS_FOLDER + folder + "/" + FILENAME), sep=","
        )

        # 格式化时间戳
        df_locations["timestamp"] = df_locations["timestamp"].apply(format_timestamp)
        # 更新文件路径
        df_locations["timestamp"] = (
            RUNS_FOLDER
            + folder
            + RANGEIMAGE_FOLS
            + df_locations["timestamp"].astype(str)
            + ".png"
        )
        # 重命名列
        df_locations = df_locations.rename(columns={"timestamp": "file"})

        db_frames = {
            "00": range(0, 20634),
            "01": range(0, 100),
            "02": range(0, 2401),
            "03": range(0, 100),
        }
        query_frames = {
            "01": range(101, 16346),
            "03": range(101, 2299),
        }

        for index, row in df_locations.iterrows():
            # print(f"当前索引: {index}")  # 输出当前索引
            # print(f"当前索引: {row}")  # 输出当前索引
            # print(f"当前索引: {db_frames[folder]}")  # 输出当前索引
            # 整个商业区都在测试集中
            if index in db_frames[folder]:
                # 使用 pd.concat()
                df_train = pd.concat([df_train, row.to_frame().T], ignore_index=True)
            elif index in query_frames[folder]:
                df_test = pd.concat([df_test, row.to_frame().T], ignore_index=True)
        print(len(df_train["file"]))
        print(len(df_test["file"]))

    print("Number of training submaps: " + str(len(df_train["file"])))
    print("Number of non-disjoint test submaps: " + str(len(df_test["file"])))

    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(
        df_train, base_path, "training_queries_kitti.pickle", ind_nn_r=5
    )
    construct_query_dict(df_test, base_path, "test_queries_kitti.pickle", ind_nn_r=5)
