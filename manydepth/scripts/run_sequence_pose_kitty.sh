#!/bin/bash

num_show_points=(10 20 30 40 50 60 70)

for i in "${num_show_points[@]}"
do
  python run_sequence_pose_kitti.py --vo_dir_path results/vo_kitti_half_split_cityscapes_pretrain/ --dataset_path /mnt/storage/workspace/andreim/kitti/data_odometry_color/pose --save_path /mnt/storage/workspace/andreim/kitti/data_odometry_color/segmentation_scsfm/ --num_show_points $i
done