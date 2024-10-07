"""
Author: lee lizw_0304@163.com
Date: 2024-10-05 18:48:07
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-05 18:49:55
FilePath: /0924-data-bag/plot_time.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_time_difference(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 计算时间差，并乘以 1000
    time_difference = (df["pointcloud_time"] - df["closest_odom_time"]) * 1000

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(time_difference, marker="o", linestyle="-", markersize=2)
    plt.title("Time Difference between Pointcloud and Closest Odometry")
    plt.xlabel("Index")
    plt.ylabel("Time Difference (ms)")
    plt.grid()
    plt.show()


def main():
    csv_file_path = (
        "data_root_folder/02/pointcloud_odom_times.csv"  # 替换为你的 CSV 文件路径
    )
    plot_time_difference(csv_file_path)


if __name__ == "__main__":
    main()
