import bisect
import os
import json
import argparse
import sys

import cv2
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd

import torch
from torchvision import transforms
from tqdm import tqdm

from utils import load_and_preprocess_intrinsics, load_and_preprocess_image

sys.path.append(os.getcwd())

import networks
from layers import transformation_from_parameters


NUM_SHOW_POINTS = 70
ALPHA = 1.
LIMIT_1 = 10
LIMIT_2 = 30


def pose_mat2vec(mat):
    """
    Convert projection matrix to rotation.
    Args:
        mat: A transformation matrix -- [B, 3, 4]
    Returns:
        3DoF parameters in the order of rx, ry, rz -- [B, 3]
    """
    import math
    import numpy as np

    mat33 = mat[:, :3, :3]
    R = mat33[0]
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--dataset_path', type=str,
                        help='path to the dataset', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to save the segmentation dataset', required=True)
    parser.add_argument('--vo_dir_path', type=str,
                        help='path to vo results dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(os.path.join(args.save_path, 'self_supervised_labels')):
        os.makedirs(os.path.join(args.save_path, 'self_supervised_labels'))
    if not os.path.exists(os.path.join(args.save_path, 'splits')):
        os.makedirs(os.path.join(args.save_path, 'splits'))
    if not os.path.exists(os.path.join(args.save_path, 'images')):
        os.makedirs(os.path.join(args.save_path, 'images'))

    train_split_path = os.path.join(args.dataset_path, 'splits', 'train_half.txt')
    val_split_path = os.path.join(args.dataset_path, 'splits', 'val_half.txt')
    test_split_path = os.path.join(args.dataset_path, 'splits', 'test.txt')

    with open(train_split_path, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    with open(val_split_path, 'r') as f:
        val_files = [line.strip() for line in f.readlines()]
    with open(test_split_path, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]

    for sequences, split in zip([train_files, val_files, test_files], ['train.txt', 'val.txt', 'test.txt']):
        print(sequences)
        f = open(os.path.join(args.save_path, 'splits', split), 'w+')
        for seq in sequences:
            if f"{seq}.txt" not in os.listdir(args.vo_dir_path):
                continue

            df = pd.read_csv(os.path.join(args.vo_dir_path, f"{seq}.txt"), sep=" ", header=None)
            path = os.path.join(args.dataset_path, 'sequence_dataset', seq, "frame{0:06d}.png")
            pose = df.values.reshape((len(df), 3, 4))

            x = np.zeros((len(pose), 1, 4))
            x[:, :, -1] = 1

            pose = np.concatenate([pose, x], axis=1)
            pose[:, :, -1] *= 32.8

            camera_matrix = np.loadtxt(os.path.join(args.dataset_path, 'sequence_dataset', seq, 'cam.txt')).astype(np.float32)
            # camera_matrix[0, :] /= 620
            # camera_matrix[1, :] /= 288
            # print(camera_matrix)
            rvec = np.array([0., 0., 0])
            tvec = np.array([0., 0., 0])
            rvec, _ = cv2.Rodrigues(rvec)

            project_points = np.array([[0, 1.7, 3, 1]]).reshape((1, 1, 4))
            project_points_l = np.array([[-0.85, 1.7, 3, 1]]).reshape((1, 1, 4))
            project_points_r = np.array([[+0.85, 1.7, 3, 1]]).reshape((1, 1, 4))

            for i in tqdm(range(len(pose))):
                crt_pose = np.stack(np.linalg.inv(pose[i]).dot(x) for x in pose[i:])

                world_points = project_points.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
                world_points_l = project_points_l.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
                world_points_r = project_points_r.dot(crt_pose.transpose((0, 2, 1)))[0, 0]

                show_img = cv2.imread(path.format(i)).astype(np.float32) / 255.

                world_points_show = np.concatenate([
                    world_points[:NUM_SHOW_POINTS][:, :3],
                    world_points_l[:NUM_SHOW_POINTS][:, :3],
                    world_points_r[:NUM_SHOW_POINTS][:, :3]
                ])

                rvec2 = crt_pose[0][:3, :3]  # it is almost the identity matrix
                show_points = cv2.projectPoints(world_points_show.astype(np.float64), rvec2, tvec,
                                                camera_matrix, None)[0]
                show_points_l = cv2.projectPoints(world_points_l[:NUM_SHOW_POINTS][:, :3].astype(np.float64), rvec2, tvec,
                                                  camera_matrix, None)[0]
                show_points_r = cv2.projectPoints(world_points_r[:NUM_SHOW_POINTS][:, :3].astype(np.float64), rvec2, tvec,
                                                  camera_matrix, None)[0]
                show_points = show_points.astype(np.int)[:, 0]
                show_points_l = show_points_l.astype(np.int)[:, 0]
                show_points_r = show_points_r.astype(np.int)[:, 0]
                overlay = np.zeros_like(show_img)

                for it, p1, p2, p3, p4 in zip(range(len(show_points_l) - 1), show_points_l[:-1], show_points_r[:-1],
                                              show_points_l[1:], show_points_r[1:]):
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    pts = np.array([(x1, y1), (x3, y3), (x4, y4), (x2, y2)])

                    overlay = cv2.drawContours(overlay, [pts], 0, (0, 255, 0), cv2.FILLED)

                # show_img = cv2.addWeighted(overlay, ALPHA, show_img, 1, 0)

                overlay = overlay[:, :, 1]
                overlay[overlay != 0] = 1

                # cv2.imshow('img', overlay)
                # cv2.waitKey(0)

                sum_euler = np.zeros(3)
                for p1, p2 in zip(pose[i:i + NUM_SHOW_POINTS], pose[i + 1:i + NUM_SHOW_POINTS + 1]):
                    relative_pose = np.linalg.inv(p1).dot(p2)
                    relative_pose = relative_pose.reshape((1, 4, 4))
                    sum_euler += (pose_mat2vec(relative_pose) * 180 / np.pi)
                sum_euler = sum_euler[1]

                limits = [-float('inf'), -LIMIT_2, -LIMIT_1, LIMIT_1, LIMIT_2, float('inf')]
                category = bisect.bisect_right(limits, sum_euler) - 1

                relative_save_path = os.path.join('{}'.format(seq) + '_frame{0:06d}.png'.format(i))
                save_path_label = os.path.join(args.save_path, 'self_supervised_labels', relative_save_path)
                save_path_image = os.path.join(args.save_path, 'images', relative_save_path)
                cv2.imwrite(save_path_label, overlay)
                cv2.imwrite(save_path_image, show_img)
                f.write(f'{relative_save_path},{sum_euler}\n')
                # print(save_path, sum_euler, category, overlay.shape)

        f.close()

if __name__ == '__main__':
    main()