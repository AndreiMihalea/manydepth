#!/bin/bash

num_show_points=(30)

for i in "${num_show_points[@]}"
do
  python run_sequence_pose_kitti.py --vo_dir_path results/vo_kitti_half_split_cityscapes_pretrain/ --gt_vo_dir_path /mnt/storage/workspace/andreim/kitti/data_odometry_color_full/dataset/poses/ --dataset_path /mnt/storage/workspace/andreim/kitti/data_odometry_color/pose --save_path /mnt/storage/workspace/andreim/kitti/data_odometry_color/segmentation/ --num_show_points $i
done