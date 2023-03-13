# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .mono_dataset import MonoDataset


class UPBDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(UPBDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]])

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        all_filenames = []

        for dir_name in self.filenames:
            files_in_dir = os.listdir(os.path.join(self.data_path, dir_name))
            cam_file = os.path.join(self.data_path, dir_name, 'cam.txt')
            files_in_dir = [(dir_name, x) for x in files_in_dir if '.txt' not in x]
            all_filenames.extend(files_in_dir)

        self.filenames = all_filenames

        cam_data = np.loadtxt(cam_file)
        self.K = cam_data
        self.K[0, :] /= 640
        self.K[1, :] /= 288
        col = np.array([[0], [0], [0]])
        row = np.array([0, 0, 0, 1])
        self.K = np.hstack([self.K, col])
        self.K = np.vstack([self.K, row]).astype(np.float32)

    def check_depth(self):
        return False

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index]
        folder = line[0]

        frame_index = int(line[1].replace(self.img_ext, '').replace('frame', ''))

        side = None

        return folder, frame_index, side

    def get_color(self, folder, frame, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index):
        image_path = os.path.join(
            self.data_path,
            folder,
            "frame{:06d}{}".format(frame_index, self.img_ext))
        return image_path
