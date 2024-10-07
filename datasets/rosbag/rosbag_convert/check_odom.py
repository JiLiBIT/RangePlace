"""
Author: lee lizw_0304@163.com
Date: 2024-10-05 17:03:30
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-05 17:05:28
FilePath: /0930-data-bag/plot_odom.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import os
import rosbag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2


def process_bag_file(bag_file_path, target_topics):
    paired_messages = []
    odom_messages = []

    with rosbag.Bag(bag_file_path, "r") as bag:
        print("ROS Bag Info:")
        print(bag)

        # 读取消息
        for topic, msg, t in bag.read_messages(topics=target_topics):
            if topic == "/odom":
                odom_time = t.to_sec()
                odom_messages.append((odom_time, msg))  # 存储时间和对应的消息

        for topic, msg, t in bag.read_messages(topics=target_topics):
            if topic == "/pointcloud":
                pointcloud_time = t.to_sec()

                # 找到距离当前点云时间最近的里程计消息
                if odom_messages:
                    # 找到与 pointcloud_time 最近的 odom_time
                    closest_odom = min(
                        odom_messages, key=lambda x: abs(x[0] - pointcloud_time)
                    )
                    closest_odom_time, closest_odom_msg = closest_odom

                    # 将最近的里程计消息与点云消息存储
                    paired_messages.append(
                        (pointcloud_time, msg, closest_odom_msg, closest_odom_time)
                    )

    return paired_messages


def plot_filtered_odometry(paired_messages):
    filtered_locations = []
    previous_location = None  # 初始化前一个位置
    starting_position = None  # 初始化起始位置

    for (
        pointcloud_time,
        pointcloud_msg,
        closest_odom_msg,
        closest_odom_time,
    ) in paired_messages:
        current_odom_position = np.array(
            [
                closest_odom_msg.pose.pose.position.x,
                closest_odom_msg.pose.pose.position.y,
            ]
        )

        # 第一次记录起始位置
        if starting_position is None:
            starting_position = current_odom_position

        # 计算与起始位置的位移
        displacement = np.linalg.norm(current_odom_position - starting_position)

        # 如果位移为零，则跳过此位置
        if displacement < 0.05:
            continue

        # 计算与前一个位置的距离
        if (
            previous_location is not None
            and np.linalg.norm(current_odom_position - previous_location) > 20
        ):
            continue  # 如果距离大于20米，则跳过此位置

        filtered_locations.append(current_odom_position)
        previous_location = current_odom_position  # 更新前一个位置

    filtered_locations = np.array(filtered_locations)

    print("有效的过滤后位置的数量:", len(filtered_locations))

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(
        filtered_locations[:, 0],
        filtered_locations[:, 1],
        marker="o",
        linestyle="-",
        markersize=2,
    )
    plt.title("Filtered Odometry Positions")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid()
    plt.show()


def main():
    bag_file_path = "robot1.bag"  # 替换为你的 bag 文件路径
    target_topics = ["/odom", "/pointcloud"]  # 替换为你要读取的话题名称

    paired_messages = process_bag_file(bag_file_path, target_topics)

    if paired_messages:
        plot_filtered_odometry(paired_messages)
        # 如有需要，可以进行其他处理（例如保存CSV，生成深度图等）
    else:
        print("未找到配对消息。")


if __name__ == "__main__":
    main()
