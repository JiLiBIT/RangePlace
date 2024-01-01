# RangePlace

The code for our paper:  

**RangePlace: A Hierarchical Range Image Transformer for LiDAR-based Place Recognition**

Developed by [Ji Li](https://github.com/JiLiBIT).

Beijing Institute of Technology



## Dependencies

coming soon ...

## Datasets

### KITTI Odometry Dataset

### Ford Campus Dataset

Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 

```
# Generate training tuples for the KITTI Dataset
cd generating_queries/ 
python generate_training_tuples_kitti.py --dataset_root <dataset_root_path>

# Generate evaluation tuples
python generate_test_kitti.py --dataset_root <dataset_root_path>
python generate_test_ford.py --dataset_root <dataset_root_path>

```

`<dataset_root_path>` is a path to dataset root folder, e.g. `/data/kitti_datasets/`.
Before running the code, ensure you have read/write rights to `<dataset_root_path>`, as training and evaluation pickles
are saved there. 



### Dataset Structure

```
  data_root_folder (KITTI for example) follows:
  ├── 00
  │   ├── depth_map
  │     ├── 000000.png
  │     ├── 000001.png
  │     ├── 000002.png
  │     ├── ...
  │   └── pointcloud_locations.csv
  ├── 01
  ├── 02
  ├── ...
  └── 10
```



## Training 
To train the network, run:

```
cd training
python train.py --config ../config/config_kitti.txt --model_config ../models/rangplace.txt
```






## Testing

coming soon ...

## License
Our code is released under the MIT License (see LICENSE file for details).

