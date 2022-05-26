from genericpath import exists
import os
from pathlib import Path
import shutil
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


def extract_deepdeform_data(datadir, scene_name, start_frame_i=0, end_frame_i=0, step=1, train_p=0.7, val_p=0.15, test_p=0.15):
    
    if end_frame_i == 0:
        end_frame_i = len(rgb_paths) - 1

    rgb_paths = os.listdir(os.path.join(datadir, "color"))[start_frame_i:end_frame_i:step]
    depth_paths = os.listdir(os.path.join(datadir, "depth"))[start_frame_i:end_frame_i:step]
    assert len(rgb_paths) == len(depth_paths), "Unequal number of RGB and depth frames."

    get_split = lambda l1, l2, p: (l1[:int(len(l1) * p)], l2[:int(len(l2) * p)])
    splits = {
        "train": get_split(rgb_paths, depth_paths, train_p), 
        "val": get_split(rgb_paths, depth_paths, val_p), 
        "test": get_split(rgb_paths, depth_paths, test_p),
    }

    with open(datadir + "/intrinsics.txt", "r") as f:
        transform_matrix = np.array([line.split(" ") for line in f.read().split("\n")[:-1]])
        transform_matrix = transform_matrix.astype(np.float32)
        assert transform_matrix.shape == (4, 4), "Transform shape mismatch."
        assert transform_matrix[3, :] == [0, 0, 0, 1], "Transform last row mismatch."
    
    new_dir = Path(f"./data/{scene_name}/").mkdir(parents=True, exist_ok=True)
    for split in splits:

        rgb_dir = Path(f"./{new_dir}/{split}/").mkdir(parents=True, exist_ok=True)
        d_dir = Path(f"./{new_dir}/{split}_depth/").mkdir(exist_ok=True)
        transforms = {
            "camera_angle_x": 0,
            "frames": []
        }
        for i, (rgb_p, d_p) in enumerate(zip(splits[split])):
            # Copy RGB-D files
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            shutil.copyfile(rgb_p, rgb_dir + f"rgb_{num_str}.png")
            shutil.copyfile(d_p, d_dir + f"d_{num_str}.jpg")
            # Add transform & time info
            frame_info = {
                "file_path": f"./{split}/rgb_{num_str}.png",
                "depth_file_path": f"./{split}/d_{num_str}.jpg",
                "rotation": 0,
                "time": i/int(end_frame_i/step),
                "transform_matrix": transform_matrix,
            }
            transforms["frames"].append(frame_info)
        
        with open(os.path.join(new_dir, f"transforms_{split}.json"), "w", encoding='utf-8') as f:
            json.dump(transforms, f, ensure_ascii=False, indent=4)


def load_deepdeform_data(basedir):

    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            # load the per-frame file-locations, times and homogeneous transforms
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]

        imgs = []
        poses = []
        times = []
        # if s=='train' or testskip==0:
        #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
        # else:
        skip = testskip
            
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            # if times are not given, assume frames are distributed uniformly over time
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
            times.append(cur_time)

        # TODO J: Verstehe nicht warum alle time==0, dann gibt es zu time=0 verschiedene Bilder 
        # -> Nein, nur das erste frame hat time=0 (times ist eine list mit allen frames)
        assert times[0] == 0, "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)
    
    # prepare indices of train/val/test splits
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    # TODO P: Ist das richtig so???
    # relationship between the AOV and focal length: https://en.wikipedia.org/wiki/Angle_of_view
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])      # horizontal AOV
    focal = .5 * W / np.tan(.5 * camera_angle_x)        # focal length

    # set the (novel) poses that are used to render novel views. Take them from the file, if given, else compute them from a sphere.
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split
