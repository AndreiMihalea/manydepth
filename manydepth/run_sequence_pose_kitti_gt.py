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

from manydepth.kitti_utils import read_calib_file
from utils import load_and_preprocess_intrinsics, load_and_preprocess_image, pose_mat2vec

sys.path.append(os.getcwd())

import networks
from layers import transformation_from_parameters


ALPHA = 1.
LIMIT_1 = 10
LIMIT_2 = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--dataset_path', type=str,
                        help='path to the dataset', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to save the segmentation dataset', required=True)
    parser.add_argument('--vo_dir_path', type=str,
                        help='path to vo results dir', required=True)
    parser.add_argument('--num_show_points', type=int,
                        help='number of points to show in the trajectory', default=70)
    parser.add_argument('--scaling_factor', type=float, default=30.51,  # 30.51 is the computed value
                        help='scaling factor for pose', required=False)
    return parser.parse_args()


def main():
    args = parse_args()

    num_show_points = args.num_show_points

    supervised_labels_path = f'self_supervised_labels_{num_show_points}'

    if not os.path.exists(os.path.join(args.save_path, supervised_labels_path)):
        os.makedirs(os.path.join(args.save_path, supervised_labels_path))
    if not os.path.exists(os.path.join(args.save_path, 'splits')):
        os.makedirs(os.path.join(args.save_path, 'splits'))
    if not os.path.exists(os.path.join(args.save_path, 'images')):
        os.makedirs(os.path.join(args.save_path, 'images'))

    train_split_path = os.path.join(args.dataset_path, 'splits', 'train.txt')
    val_split_path = os.path.join(args.dataset_path, 'splits', 'val.txt')
    test_split_path = os.path.join(args.dataset_path, 'splits', 'test.txt')

    with open(train_split_path, 'r') as f:
        train_files = {line.strip()[:-2]: f'train_{num_show_points}' for line in f.readlines()}
    with open(val_split_path, 'r') as f:
        val_files = {line.strip()[:-2]: f'val_{num_show_points}' for line in f.readlines()}
    with open(test_split_path, 'r') as f:
        test_files = {line.strip()[:-2]: f'test_{num_show_points}' for line in f.readlines()}

    all_files = {**train_files, **val_files, **test_files}

    # print(all_files)

    sequences = os.listdir(args.vo_dir_path)
    sequences = set([seq.split('.')[0] for seq in sequences])

    fds = {}

    for data_split in [f'train_{num_show_points}.txt', f'val_{num_show_points}.txt', f'test_{num_show_points}.txt']:
        fds[data_split.split('.')[0]] = open(os.path.join(args.save_path, 'splits', data_split), 'w+')

    for seq in sequences:
        print(os.path.join(args.vo_dir_path, f'{seq}.txt'))
        df = pd.read_csv(os.path.join(args.vo_dir_path, f'{seq}.txt'), sep=' ', header=None)
        path = os.path.join(args.dataset_path, 'sequences', seq, 'image_2', '{0:06d}.png')
        pose = df.values.reshape((len(df), 3, 4))

        x = np.zeros((len(pose), 1, 4))
        x[:, :, -1] = 1

        pose = np.concatenate([pose, x], axis=1)
        # pose[:, :, -1] *= args.scaling_factor

        camera_matrix, _ = read_calib_file(os.path.join(args.dataset_path, 'sequences', seq, 'calib.txt'))
        camera_matrix = camera_matrix[:3, :3]
        # camera_matrix[0, :] /= 620
        # camera_matrix[1, :] /= 288
        # print(camera_matrix)
        rvec = np.array([0., 0., 0])
        tvec = np.array([0., 0., 0])
        rvec, _ = cv2.Rodrigues(rvec)

        project_points = np.array([[0, 1.7, 3, 1]]).reshape((1, 1, 4))
        project_points_l = np.array([[-0.8, 1.65, 1.68, 1]]).reshape((1, 1, 4))
        project_points_r = np.array([[+0.8, 1.65, 1.68, 1]]).reshape((1, 1, 4))

        for i in tqdm(range(len(pose))):
            crt_pose = np.stack(np.linalg.inv(pose[i]).dot(x) for x in pose[i:])

            world_points = project_points.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
            world_points_l = project_points_l.dot(crt_pose.transpose((0, 2, 1)))[0, 0]
            world_points_r = project_points_r.dot(crt_pose.transpose((0, 2, 1)))[0, 0]

            show_img = cv2.imread(path.format(i))

            world_points_show = np.concatenate([
                world_points[:num_show_points][:, :3],
                world_points_l[:num_show_points][:, :3],
                world_points_r[:num_show_points][:, :3]
            ])

            rvec2 = crt_pose[0][:3, :3]  # it is almost the identity matrix
            show_points = cv2.projectPoints(world_points_show.astype(np.float64), rvec2, tvec,
                                            camera_matrix, None)[0]
            show_points_l = cv2.projectPoints(world_points_l[:num_show_points][:, :3].astype(np.float64), rvec2, tvec,
                                              camera_matrix, None)[0]
            show_points_r = cv2.projectPoints(world_points_r[:num_show_points][:, :3].astype(np.float64), rvec2, tvec,
                                              camera_matrix, None)[0]
            show_points = show_points.astype(np.int)[:, 0]
            show_points_l = show_points_l.astype(np.int)[:, 0]
            show_points_r = show_points_r.astype(np.int)[:, 0]
            overlay = np.zeros_like(show_img)

            try:
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

                # sum_euler accumulates ALzi49NaR%3b
                # differences between the y rotations of two adjacent poses along a trajectory
                sum_euler = np.zeros(3)
                for p1, p2 in zip(pose[i:i + num_show_points], pose[i + 1:i + num_show_points + 1]):
                    relative_pose = np.linalg.inv(p1).dot(p2)
                    relative_pose = relative_pose.reshape((1, 4, 4))
                    sum_euler += (pose_mat2vec(relative_pose) * 180 / np.pi)
                sum_euler = sum_euler[1]

                # diff_euler calculates the difference in y rotations between the last and first points of a trajectory
                p1 = pose[i]
                p2 = pose[i:i + num_show_points + 1][-1]
                relative_pose = np.linalg.inv(p1).dot(p2)
                relative_pose = relative_pose.reshape((1, 4, 4))
                diff_euler = (pose_mat2vec(relative_pose) * 180 / np.pi)[1]

                limits = [-float('inf'), -LIMIT_2, -LIMIT_1, LIMIT_1, LIMIT_2, float('inf')]
                category = bisect.bisect_right(limits, sum_euler) - 1

                relative_save_path = os.path.join('{}'.format(seq) + '_frame{0:06d}.png'.format(i))
                save_path_label = os.path.join(args.save_path, supervised_labels_path, relative_save_path)
                save_path_image = os.path.join(args.save_path, 'images', relative_save_path)

                # print(save_path_image, save_path_label)

                # cv2.imwrite(save_path_label, overlay)
                # cv2.imwrite(save_path_image, show_img)
                fds[all_files[f'{int(seq)} {i}']].write(f'{relative_save_path},{sum_euler},{diff_euler}\n')
            except:
                continue

    for fd in fds:
        fds[fd].close()


if __name__ == '__main__':
    main()