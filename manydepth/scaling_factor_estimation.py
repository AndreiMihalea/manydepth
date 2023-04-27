# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import random

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

import networks
from inverse_warp_utils import get_factor
from options import MonodepthOptions
from utils import load_and_preprocess_image, load_and_preprocess_intrinsics
from layers import transformation_from_parameters, disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--dataset_path', type=str,
                        help='path to the dataset to take images from', required=True)
    parser.add_argument('--intrinsics_json_path', type=str,
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    parser.add_argument('--num_samples', type=int, default=200,
                        help='number of samples to take the scaling factor from')
    return parser.parse_args()


def compute_scaling_factor(args):
    """
    Function that computes the scaling factor by sampling from different images in the dataset
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)
    encoder = networks.ResnetEncoderMatching(18, False,
                                             input_width=encoder_dict['width'],
                                             input_height=encoder_dict['height'],
                                             adaptive_bins=True,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'],
                                             depth_binning='linear',
                                             num_depth_bins=96)

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(os.path.join(args.model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    print("   Loading pose network")
    pose_enc_dict = torch.load(os.path.join(args.model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(args.model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    all_frames = []
    dataset_path = args.dataset_path
    sequences = os.listdir(dataset_path)
    for seq in sequences:
        seq_images = os.listdir(os.path.join(dataset_path, seq, 'image_2'))
        seq_images = [os.path.join(dataset_path, seq, 'image_2', img_name) for img_name in seq_images]
        all_frames.extend(seq_images)

    random.shuffle(all_frames)
    samples = all_frames[:args.num_samples]

    scaling_factor = torch.zeros(1)

    for sample in samples:
        # Load input data
        target_image_path = sample
        frame_name = target_image_path.split('/')[-1]
        frame_name = frame_name.split('.')[0]
        next_frame_value = int(frame_name) + 1
        source_image_path = target_image_path.replace(frame_name, '{:06d}'.format(next_frame_value))
        input_image, original_size = load_and_preprocess_image(target_image_path,
                                                               resize_width=encoder_dict['width'],
                                                               resize_height=encoder_dict['height'])

        source_image, _ = load_and_preprocess_image(source_image_path,
                                                    resize_width=encoder_dict['width'],
                                                    resize_height=encoder_dict['height'])

        K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                                 resize_width=encoder_dict['width'],
                                                 resize_height=encoder_dict['height'])

        with torch.no_grad():

            # Estimate poses
            pose_inputs = [source_image, input_image]
            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = pose_dec(pose_inputs)
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

            if args.mode == 'mono':
                pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
                source_image *= 0

            # Estimate depth
            output, lowest_cost, _ = encoder(current_image=input_image,
                                             lookup_images=source_image.unsqueeze(1),
                                             poses=pose.unsqueeze(1),
                                             K=K,
                                             invK=invK,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'])

            output = depth_decoder(output)

            sigmoid_output = output[("disp", 0)]
            sigmoid_output_resized = torch.nn.functional.interpolate(
                sigmoid_output, original_size, mode="bilinear", align_corners=False)
            sigmoid_output_resized = sigmoid_output_resized.cpu()

            print(original_size, 'og')

            # K_full_size = np.array([[0.58, 0, 0.5],
            #                         [0, 1.92, 0.5],
            #                         [0, 0, 1]]).astype(np.float32)

            # K_full_size = K[0, :3, :3].cpu().numpy()
            #
            # K_full_size[0, :] *= 4
            # K_full_size[1, :] *= 4

            K_full_size = np.array([[481.94055176, 0., 407.07849121],
                                    [0., 489.43386841, 126.10430145],
                                    [0., 0., 1.]]
                                   ).astype(np.float32)
            K_full_size[0, :] *= original_size[1] / 832
            K_full_size[1, :] *= original_size[0] / 256
            print(K_full_size)

            _, depth = disp_to_depth(sigmoid_output_resized, 0.1, 100.)

            scaling_factor += get_factor(depth, K_full_size)

    print(scaling_factor / args.num_samples)
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    compute_scaling_factor(args)
