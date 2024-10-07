import os
import rosbag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import cv2


def range_projection_HDL16E(
    current_vertex, fov_up=15.5, fov_down=-15.5, proj_H=16, proj_W=900, max_range=100
):
    """Project a pointcloud into a spherical projection, range image."""
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[
        (depth > 0) & (depth < max_range)
    ]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_x = np.clip(proj_x, 0, proj_W - 1)

    proj_y = np.floor(proj_y).astype(np.int32)
    proj_y = np.clip(proj_y, 0, proj_H - 1)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full(
        (proj_H, proj_W), -1, dtype=np.float32
    )  # [H,W] range (-1 is no data)
    proj_vertex = np.full(
        (proj_H, proj_W, 4), -1, dtype=np.float32
    )  # [H,W] index (-1 is no data)
    proj_idx = np.full(
        (proj_H, proj_W), -1, dtype=np.int32
    )  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]
    ).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx


def process_bag_file(bag_file_path, target_topics, output_dir):
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


def save_csv(output_dir, transformed_locations):
    original_df = pd.DataFrame(
        transformed_locations, columns=["timestamp", "northing", "easting"]
    )
    original_df.to_csv(
        os.path.join(output_dir, "pointcloud_locations_origin.csv"), index=False
    )

    df = original_df[["northing", "easting"]].copy()
    df.set_index(pd.RangeIndex(start=0, stop=len(df), step=1), inplace=True)
    df.rename_axis("timestamp", inplace=True)
    df.to_csv(
        os.path.join(output_dir, "pointcloud_locations.csv"), index_label="timestamp"
    )

    print(
        f"Point cloud locations saved to '{output_dir}/pointcloud_locations_origin.csv' and '{output_dir}/pointcloud_locations.csv' with {len(transformed_locations)} entries."
    )


def generate_depth_maps(output_dir, paired_messages):
    depth_map_dir = os.path.join(output_dir, "depth_map")  # 改为 depth_map 文件夹
    os.makedirs(depth_map_dir, exist_ok=True)

    for idx, (pointcloud_time, pointcloud_msg, odom_msg, odom_time) in enumerate(
        paired_messages
    ):
        current_vertex = pc2.read_points(
            pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        current_vertex = np.array(list(current_vertex))

        proj_range, proj_vertex, proj_idx = range_projection_HDL16E(current_vertex)

        # 文件名格式为6位数字，不足补零
        depth_map_filename = os.path.join(depth_map_dir, f"{idx:06d}.png")
        plt.imsave(
            depth_map_filename, proj_range, cmap="gray", vmin=0, vmax=np.max(proj_range)
        )

        print(f"Depth map saved as: {depth_map_filename}")


def play_depth_maps_1080P(depth_map_dir):
    """Play depth maps at 1080P resolution."""
    depth_map_files = sorted(
        [f for f in os.listdir(depth_map_dir) if f.endswith(".png")]
    )

    for depth_map_file in depth_map_files:
        depth_map_path = os.path.join(depth_map_dir, depth_map_file)
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

        # 将深度图放大到 1080P
        depth_map_resized = cv2.resize(depth_map, (1920, 1080))

        # 显示图像
        cv2.imshow("Depth Map", depth_map_resized)
        if cv2.waitKey(10):  # 等待10毫秒
            continue

    cv2.destroyAllWindows()


def play_depth_maps_width(depth_map_dir):
    """Play depth maps at a width of 1920 pixels while maintaining aspect ratio."""
    depth_map_files = sorted(
        [f for f in os.listdir(depth_map_dir) if f.endswith(".png")]
    )

    for depth_map_file in depth_map_files:
        depth_map_path = os.path.join(depth_map_dir, depth_map_file)
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

        # 获取原始尺寸
        original_height, original_width = depth_map.shape

        # 计算等比缩放的高度
        new_width = 1920
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)

        # 将深度图等比缩放
        depth_map_resized = cv2.resize(depth_map, (new_width, new_height))

        # 显示图像
        cv2.imshow("Depth Map", depth_map_resized)
        if cv2.waitKey(33):  # 等待33毫秒
            continue

    cv2.destroyAllWindows()


def main():
    output_dir = "data_root_folder/02"  # 替换为你的存储文件路径的编号
    os.makedirs(output_dir, exist_ok=True)

    bag_file_path = "robot1.bag"  # 替换为你的 bag 文件路径
    target_topics = ["/odom", "/pointcloud"]  # 替换为你要读取的话题名称

    paired_messages = process_bag_file(bag_file_path, target_topics, output_dir)

    if paired_messages:
        first_odom_msg = paired_messages[0][2]
        first_odom_position = np.array(
            [
                first_odom_msg.pose.pose.position.x,
                first_odom_msg.pose.pose.position.y,
                first_odom_msg.pose.pose.position.z,
            ]
        )

        transformed_locations = []
        filtered_paired_messages = []  # 新建一个列表用于保存有效的配对消息
        previous_location = None  # 初始化前一个位置

        for pointcloud_time, pointcloud_msg, odom_msg, odom_time in paired_messages:
            current_odom_position = np.array(
                [
                    odom_msg.pose.pose.position.x,
                    odom_msg.pose.pose.position.y,
                    odom_msg.pose.pose.position.z,
                ]
            )

            local_position = current_odom_position - first_odom_position

            if np.allclose(local_position, 0):
                local_position = np.array([0, 0])

            # 计算当前与前一个位置的距离
            if (
                previous_location is not None
                and np.linalg.norm(local_position - previous_location) > 20
            ):
                continue  # 如果距离大于20米，则跳过此位置

            if np.linalg.norm(local_position) > 0.05:
                transformed_locations.append(
                    (pointcloud_time, local_position[1], local_position[0])
                )
                filtered_paired_messages.append(
                    (pointcloud_time, pointcloud_msg, odom_msg, odom_time)
                )  # 保存有效的配对消息
                previous_location = local_position  # 更新前一个位置

        # 更新 paired_messages 为过滤后的有效消息
        paired_messages = filtered_paired_messages

        df = pd.DataFrame(
            paired_messages,
            columns=[
                "pointcloud_time",
                "pointcloud_msg",
                "closest_odom_msg",
                "closest_odom_time",
            ],
        )

        selected_columns = df[["pointcloud_time", "closest_odom_time"]].copy()

        # 保存为CSV文件
        csv_file_path = os.path.join(output_dir, "pointcloud_odom_times.csv")
        selected_columns.to_csv(csv_file_path, index=False)
        print(f"数据已保存到 {csv_file_path}")

        # 保存位置信息
        save_csv(output_dir, transformed_locations)
        generate_depth_maps(output_dir, paired_messages)
        play_depth_maps_width(os.path.join(output_dir, "depth_map"))  # 播放深度图
    else:
        print("No paired messages found.")


if __name__ == "__main__":
    main()
