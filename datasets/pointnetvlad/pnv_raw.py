import numpy as np
import os
import cv2
from datasets.base_datasets import PointCloudLoader, RangeImageLoader


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        pc = np.fromfile(file_path, dtype=np.float64)
        pc = np.float32(pc)
        # coords are within -1..1 range in each dimension
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        return pc

class PNVRangeImageLoader(RangeImageLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_ri(self, file_pathname: str) -> np.ndarray:
        ri = np.array(cv2.imread(file_pathname, cv2.IMREAD_GRAYSCALE))
        ri = ri.astype('float32')
        #depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor)
        ri = np.expand_dims(ri, axis=0)

        return ri