import os
import json
import argparse
import sys

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from tqdm import tqdm

from utils import load_and_preprocess_image

sys.path.append(os.getcwd())

import networks
from layers import transformation_from_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for ManyDepth models.')

    parser.add_argument('--dataset_path', type=str,
                        help='path to the dataset', required=True)
    parser.add_argument("--output_path", type=str,
                        help="Output directory for saving predictions in a big 3D numpy file")
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    parser.add_argument("--img_exts", default=['png', 'jpg', 'bmp'],
                        nargs='*', type=str, help="images extensions to glob")
    return parser.parse_args()


def main():
    args = parse_args()

    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading encoder")
    encoder_dict = torch.load(os.path.join(args.model_path, "encoder.pth"), map_location=device)

    print("Loading pose network")
    pose_enc_dict = torch.load(os.path.join(args.model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(args.model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2).to(device)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2).to(device)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    pose_enc.eval()
    pose_dec.eval()

    sequences = os.listdir(args.dataset_path)

    with torch.no_grad():
        for seq in sequences:
            for side in ['image_2', 'image_3']:
                image_dir = os.path.join(args.dataset_path, seq, side)
                test_files = os.listdir(image_dir)
                test_files = [os.path.join(image_dir, file_name) for file_name in test_files if
                              any(file_name.endswith(ext) for ext in args.img_exts)]

                test_files.sort()
                print('{} files to test'.format(len(test_files)))

                n = len(test_files)

                global_pose = torch.eye(4)
                poses = [global_pose[0:3, :].reshape(1, 12)]

                for it in tqdm(range(n - 1)):
                    try:
                        source_image, _ = load_and_preprocess_image(test_files[it],
                                                                    resize_width=encoder_dict['width'],
                                                                    resize_height=encoder_dict['height'])
                        input_image, _ = load_and_preprocess_image(test_files[it + 1],
                                                                   resize_width=encoder_dict['width'],
                                                                   resize_height=encoder_dict['height'])
                    except:
                        print(f"something happened with {test_files[it + 1]}")
                        continue
                    pose_inputs = [source_image, input_image]
                    pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                    axisangle, translation = pose_dec(pose_inputs)
                    pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)[0]

                    global_pose = torch.matmul(global_pose, pose.cpu())
                    poses.append(global_pose[0:3, :].reshape(1, 12))

                poses = np.concatenate(poses, axis=0)

                filename = os.path.join(args.output_path, f'{seq}.{side}.txt')
                np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()