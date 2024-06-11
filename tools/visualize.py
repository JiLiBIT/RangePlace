import sys
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('..')


# define the utils
def show_image(image, title=''):
    # image is [H, W, 1]
    assert image.shape[2] == 1
    # image = torch.clip(image * 255, 0, 255).int()
    image = np.asarray(image)
    # image = np.asarray(Image.fromarray(np.float32(image)).resize((64, 900)))
    plt.imshow(image)
    # plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_, arch='swin_mae'):
    # build model
    model = getattr(swin_mae, arch)()
    model = model.cpu()
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

# 读取二进制点云文件
def read_bin_point_cloud(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points
def visualize_point_cloud(points, point_size=0.1):
    # 提取坐标和颜色信息
    coordinates = points[:, :3]
    distances = np.linalg.norm(coordinates, axis=1)  # 计算点到原点的距离
    colors = distances / np.max(distances)  # 根据距离进行归一化，得到颜色信息

    # 创建3D图形窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors, cmap='viridis', s=point_size)

    # 关闭坐标轴
    ax.set_axis_off()
    ax.set_box_aspect([10, 10, 10])  # 调整比例以放大z轴
    ax.auto_scale_xyz([-10, 10], [-10, 10], [-10, 10])  # 根据需要调整坐标轴范围
    plt.savefig('/home/liji/PC.png', bbox_inches='tight', pad_inches=0)
    # 显示图形
    plt.show()

def run_one_image(x):
    # make it a batch-like
    x = x.unsqueeze(dim=0)


    plt.rcParams['figure.figsize'] = [12, 6]

    plt.subplot(1, 1, 1)
    show_image(x[0], "original")
    plt.savefig('/home/liji/RI.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
if __name__ == '__main__':
    # 读取图像
    img_root = '/media/liji/T7/Dataset/OT/pretrain/00/depth_map'
    img_name = '001802.png'
    bin_root = '/media/liji/T71/Dataset/KITTI_odometry_benchmark/data_odometry_velodyne/dataset/sequences/00/velodyne/001802.bin'
    # img = np.array(cv2.imread(img_root + img_name,
    #                         cv2.IMREAD_GRAYSCALE))
    # img = img.astype('float32')
    # input_tensor = torch.as_tensor(img)

    img = Image.open(os.path.join(img_root, img_name))
    # img = img.resize((224, 224))
    img = np.asarray(img)
    input_tensor = torch.as_tensor(img)
    input_tensor = torch.unsqueeze(input_tensor, dim=2).cpu()
    print("input",input_tensor.shape)
    assert input_tensor.shape == (64, 900, 1)

    # # 读取模型
    # chkpt_dir = '/home/liji/HIOT/HiOT-master/pretrain/weights/checkpoint-98.pth'
    # model_mae = prepare_model(chkpt_dir, 'swin_mae')
    # print('Model loaded.')
    # plot_dir = '/home/liji/HIOT/HiOT-master/pretrain/'

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(3407)
    print('MAE with pixel reconstruction:')
    run_one_image(input_tensor)
    point_cloud = read_bin_point_cloud(bin_root)
    visualize_point_cloud(point_cloud, point_size=0.1)
