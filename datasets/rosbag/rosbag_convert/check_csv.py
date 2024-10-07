"""
Author: lee lizw_0304@163.com
Date: 2024-10-05 18:48:07
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-05 18:49:38
FilePath: /0924-data-bag/plot_csv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

"""
Author: lee lizw_0304@163.com
Date: 2024-10-05 17:43:58
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-05 18:23:23
FilePath: /0930-data-bag/plot_csv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

"""
Author: lee lizw_0304@163.com
Date: 2024-10-05 17:14:02
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-05 17:32:31
FilePath: /0924-data-bag/plot_csv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_pointcloud_locations(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 提取 northing 和 easting 数据
    northing = df["northing"]
    easting = df["easting"]

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.scatter(easting, northing, c="blue", marker="o", s=10)  # 使用散点图
    plt.title("Point Cloud Locations")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.grid()
    plt.axis("equal")  # 设置坐标轴等比
    plt.show()


# 示例用法
if __name__ == "__main__":
    csv_file_path = (
        "data_root_folder/02/pointcloud_locations.csv"  # 替换为实际的 CSV 文件路径
    )
    plot_pointcloud_locations(csv_file_path)
