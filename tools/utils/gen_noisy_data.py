import numpy as np
from PIL import Image
from utils import load_files
import os

def gen_noisy_data(clean_folder, dst_folder, normalize=False):
    # 读取图像

    dst_folder = os.path.join(dst_folder, 'noisy')
    try:
        os.stat(dst_folder)
        print('generating noisy data in: ', dst_folder)
    except:
        print('creating new noisy folder: ', dst_folder)
        os.mkdir(dst_folder)
    clean_paths = load_files(clean_folder)
    print(clean_folder)

    print(len(clean_paths))
    for idx in range(len(clean_paths)):
        image = Image.open(clean_paths[idx])

        # 将图像转换为NumPy数组
        image_array = np.array(image)

        # 获取图像的尺寸
        height, width = 64, 900

        # 设置高斯噪声的参数
        mean = 0  # 均值
        stddev = 10  # 标准差

        # 遍历图像的每个像素
        for i in range(height):
            for j in range(width):
                # 以八分之一的概率应用高斯噪声
                if np.random.random() < 0.125:
                    # 生成一个正态分布的随机数
                    noise = np.random.normal(mean, stddev)
                    
                    # 减去高斯噪声
                    image_array[i, j] -= noise

        # 将处理后的数组转换回图像
        processed_image = Image.fromarray(image_array)
        

        # 保存处理后的图像
        dst_path = os.path.join(dst_folder, str(idx).zfill(6))
        filename = dst_path + ".png"
        processed_image.save(filename)


if __name__ == '__main__':
    for idx in range(2, 10):
        #scan_folder ='/home/liji/workshop/OverlapTransformer-master/demo/scans/'#'path_to_source_bin'
        clean_folder ='F:/Dataset/OT/data_root_folder/' + str(idx).zfill(2) + 'depth_map/'#'path_to_source_bin'
        #scan_folder ='/media/liji/新加卷/Dataset/KITTI_odometry_benchmark/data_odometry_velodyne/dataset/sequences/01/velodyne' #'path_to_source_bin'
        dst_folder ='F:/Dataset/OT/data_root_folder/' + str(idx).zfill(2) #'path_to_saved_png'
        #dst_folder ='/media/liji/新加卷/Dataset/OT/data_root_folder/01'  #'path_to_saved_png'
        depth_data = gen_noisy_data(clean_folder, dst_folder)
