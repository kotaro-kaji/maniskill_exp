import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from robo_manip_baselines.common import DataKey, DataManager

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("out_video_filename", type=str)
parser.add_argument("in_npz_dir", type=str)
parser.add_argument("--column_num", type=int, required=False)
parser.add_argument("--envs", nargs="*", required=False)
args = parser.parse_args()

front_images = []
img_l = 0
if args.envs is None:
    args.envs = []
    env_paths = glob.glob(os.path.join(args.in_npz_dir, "env*"))
    env_paths.sort()
    for env_path in env_paths:
        args.envs.append(os.path.split(env_path)[-1])
if args.column_num is None:
    args.column_num = len(args.envs)
row_num = int(np.ceil(len(args.envs) / args.column_num))

for e in args.envs:
    files = glob.glob(os.path.join(args.in_npz_dir, f"{e}/*.npz"))
    files.sort()

    data_manager = DataManager(env=None)
    print(f"[tile_teleop_videos] Load a npz file: {files[0]}")
    data_manager.load_data(files[0])

    front_images.append(
        data_manager.get_data_seq(DataKey.get_rgb_image_key("front"))[:, ::2, ::2, :]
    )
    img_l = front_images[-1].shape[0] if img_l < front_images[-1].shape[0] else img_l

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
video = cv2.VideoWriter(
    args.out_video_filename,
    fourcc,
    60.0,
    (
        (front_images[0].shape[2] + 10) * args.column_num,
        (front_images[0].shape[1] + 10) * row_num,
    ),
)

white_img = np.full(
    (front_images[0].shape[1] + 10, front_images[0].shape[2] + 10, 3),
    255,
    dtype=np.uint8,
)
im_list = []
for i in range(row_num * args.column_num):
    im_list.append(white_img)
im_array = np.array(im_list)

for i in range(img_l):
    for j in range(len(args.envs)):
        if front_images[j].shape[0] > i:
            image_pil = Image.fromarray(front_images[j][i])
            im_array[j] = ImageOps.expand(image_pil, border=5, fill="white")
        else:
            image_pil = Image.fromarray(front_images[j][front_images[j].shape[0] - 1])
            im_array[j] = ImageOps.expand(image_pil, border=5, fill="white")

    im_tile = cv2.vconcat(
        [
            cv2.hconcat(im_array_h)
            for im_array_h in im_array.reshape(
                row_num,
                args.column_num,
                white_img.shape[0],
                white_img.shape[1],
                white_img.shape[2],
            )
        ]
    )

    video.write(cv2.cvtColor(im_tile, cv2.COLOR_RGB2BGR))

video.release()
print(f"[tile_teleop_videos] Save a video: {args.out_video_filename}")
