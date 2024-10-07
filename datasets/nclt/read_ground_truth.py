# !/usr/bin/python
#
# 示例代码用于读取和绘制真实轨迹数据。
#
# 注意：真实轨迹数据的采样频率约为 100 Hz。为了生成这个高频率的真实轨迹，使用了 SLAM 解决方案。在SLAM图中的节点不是以 100 Hz的频率添加的，而是每隔约 8 米添加一次。在SLAM图节点# 之间，使用里程计进行插值，以提供高频率的真实轨迹。如果需要精确的姿态（例如，用于累积点云），建议仅使用与SLAM图中节点对应的真实轨迹姿态。这可以通过检查协方差文件中的时间戳来实现。
#
# 调用方式：
#
#   python read_ground_truth.py groundtruth.csv covariance.csv


import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


def main(args):

    if len(args) < 3:
        print("Please specify ground truth and covariance files")
        return 1

    gt = np.loadtxt(sys.argv[1], delimiter=",")
    cov = np.loadtxt(sys.argv[2], delimiter=",")

    t_cov = cov[:, 0]  # 所有的行，第一列

    # 注意：插值并不是必需的，这只是为了方便
    interp = scipy.interpolate.interp1d(
        gt[:, 0], gt[:, 1:], kind="nearest", axis=0
    )  # 创建插值函数
    pose_gt = interp(t_cov)  # 使用协方差中的时间戳进行插值

    # NED (North, East Down)
    x = pose_gt[:, 0]
    y = pose_gt[:, 1]
    z = pose_gt[:, 2]

    r = pose_gt[:, 3]
    p = pose_gt[:, 4]
    h = pose_gt[:, 5]

    plt.figure()
    plt.scatter(y, x, 1, c=-z, linewidth=0)  # Note Z points down
    plt.axis("equal")
    plt.title("Ground Truth Position of Nodes in SLAM Graph")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.colorbar()

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
