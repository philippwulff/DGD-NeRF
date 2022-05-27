import os
from pathlib import Path
import shutil
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


def extract_deepdeform_data(datadir, scene_name, start_frame_i=0, end_frame_i=None, step=1, train_p=0.7, val_p=0.15, test_p=0.15):
    """Converts a given sequence from the DeepDeform dataset to the required format.
    Download DeepDeform at: https://github.com/AljazBozic/DeepDeform

    Args:
        datadir (str): The path leading to the sequence directory, e.g. path/to/deepdeform/seq000/
        scene_name (str): The name for the created scene folder.
        start_frame_i (int, optional): Index of the first image for the new scene. Defaults to 0.
        end_frame_i (int, optional): Index of the last image for the new scene. Defaults to None.
        step (int, optional): Set to 1 to have the same frame-rate as in DeepDeform and >1 for a slower frame-rate. 
            Defaults to 1.
        train_p (float, optional): Training set fraction. Defaults to 0.7.
        val_p (float, optional): Validation set fraction. Defaults to 0.15.
        test_p (float, optional): Test set fraction. Defaults to 0.15.
    """

    rgb_paths = sorted(os.listdir(os.path.join(datadir, "color")))[start_frame_i:end_frame_i:step]
    depth_paths = sorted(os.listdir(os.path.join(datadir, "depth")))[start_frame_i:end_frame_i:step]
    if not end_frame_i:
        end_frame_i = len(rgb_paths) - 1
    assert len(rgb_paths) == len(depth_paths), "Unequal number of RGB and depth frames."
    
    frames = [{"rgb": os.path.join(datadir, "color", rgb), "d": os.path.join(datadir, "depth", d), "t": t} 
                    for rgb, d, t in zip(rgb_paths, depth_paths, np.linspace(0, 1, len(rgb_paths)))]
    np.random.shuffle(frames)

    # Create train-val-test splits from shuffled frames and then sort by time
    splits = {
        "train": sorted(frames[:int(len(frames)*train_p)], key=lambda x: x["t"]), 
        "val": sorted(frames[int(len(frames)*train_p):int(len(frames)*(train_p+val_p))], key=lambda x: x["t"]), 
        "test": sorted(frames[int(len(frames)*(train_p+val_p)):], key=lambda x: x["t"]),
    }
    print(f"Creating {int(train_p*100)}-{int(val_p*100)}-{int(test_p*100)}-Split with {len(splits['train'])}-{len(splits['val'])}-{len(splits['test'])} images.")

    with open(datadir + "/intrinsics.txt", "r") as f:
        transform_matrix = np.array([line.split(" ") for line in f.read().split("\n")[:-1]])
        transform_matrix = transform_matrix.astype(np.float32)
        assert transform_matrix.shape == (4, 4), "Transform shape mismatch."
        assert all(transform_matrix[3, :] == [0, 0, 0, 1]), "Transform last row mismatch."
    
    for s in splits:
        rgb_dir = Path(f"./data/{scene_name}/{s}/")
        d_dir = Path(f"./data/{scene_name}/{s}_depth/")
        rgb_dir.mkdir(parents=True, exist_ok=True)
        d_dir.mkdir(exist_ok=True)
        transforms = {
            "camera_angle_x": 0.6911112070083618,        # The FOV in x dimension
            "frames": []
        }
        for i, frame in enumerate(splits[s]):
            # Copy RGB and depth files
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            shutil.copyfile(frame["rgb"],  f"{rgb_dir}/rgb_{num_str}.png")
            shutil.copyfile(frame["d"], f"{d_dir}/d_{num_str}.jpg")
            # Add transform and time info
            frame_info = {
                "file_path": f"./{s}/rgb_{num_str}",        # without .png
                "depth_file_path": f"./{s}/d_{num_str}",    # without .jpg
                "rotation": 0,
                "time": frame["t"],
                "transform_matrix": transform_matrix.tolist(),
            }
            transforms["frames"].append(frame_info)
        
        with open(os.path.join(f"./data/{scene_name}", f"transforms_{s}.json"), "w", encoding='utf-8') as f:
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

        # DeepDeform does not have multiple views at t=0
        # assert times[0] == 0, "Time must start at 0"

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


if __name__ == "__main__":
    print("FOR DEBUGGING")
    extract_deepdeform_data("/mnt/raid/kirwul/deepdeform/train/seq120", "office")
