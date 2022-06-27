import os
from pathlib import Path
import shutil
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import math

from utils.load_blender import trans_t, rot_phi, rot_theta


SCENE_OBJECT_DEPTH = 1.5         # Distance to the main object of the scene in meters

def pose_spherical2(alpha, beta, radius):
    """Computes camera poses on a sphere around the world coordinate origin without spherical coordinates.

    Args:
        alpha (float): Rotation about the X-axis.
        beta (float): Rotation about the Y-axis.
        radius (float): Translation in the radial (Z) direction.

    Returns:
        c2w: 4x4 Tensor. The camera-to-world homogeneous transformation matrix.
    """
    c2w = trans_t(radius)
    c2w = rot_phi(alpha/180.*np.pi) @ c2w
    c2w = rot_theta(-beta/180.*np.pi) @ c2w
    return c2w


def pose_spiral(theta, z_cam, z_cam_glob, H, W):
    """Computes camera poses that spiral out from the fixed training pose at (0, 0, z_glob) in the direction 
    of the world coordinate origin and stay inside of the training camera's visibility cone.

    Args:
        theta (float): Angle of rotation about Z_cam in degrees.
        z_cam (float): Distance in Z in camera coordinates.
        z_cam_glob (float): Distance in Z from global coordinate origin to the training camera.
        H (int): Image height.
        W (int): Image width.

    Returns:
        c2w: 4x4 Tensor.
    """
    # in camera coordinates
    lim = min(H/W, W/H)
    x_c = z_cam * lim * np.cos(theta * np.pi/180)
    y_c = z_cam * lim * np.sin(theta * np.pi/180)
    # in world coordinates 
    x_w, y_w, z_w, _ = (trans_t(z_cam_glob) @ torch.Tensor([x_c, y_c, z_cam, 1])).cpu()
    alpha = - np.arctan(y_w/z_w) * 180/np.pi     # is rotation about X_global
    beta = np.arctan(x_w/z_w) * 180/np.pi      # rotation about Y_global
    radius = np.sqrt(x_w**2 + y_w**2 + z_w**2)
    c2w = pose_spherical2(alpha, beta, radius)
    return c2w


def extract_deepdeform_data(datadir, scene_name, start_frame_i=0, end_frame_i=None, step=1, train_p=0.7, val_p=0.15, test_p=0.15):
    """Converts a given sequence from the DeepDeform dataset to the required format.
    Download DeepDeform at: https://github.com/AljazBozic/DeepDeform
    All lengths is in mm.

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
        intrinsics_matrix = np.array([line.split(" ") for line in f.read().split("\n")[:-1]])
        intrinsics_matrix = intrinsics_matrix.astype(np.float32)
        assert intrinsics_matrix.shape == (4, 4), "Intrinsics shape mismatch."
        assert all(intrinsics_matrix[3, :] == [0, 0, 0, 1]), "Intrinsics last row mismatch."
        f_x = intrinsics_matrix[0, 0]
        f_y = intrinsics_matrix[1, 1]

    transform_matrix = np.identity(4)
    transform_matrix[2, 3] = SCENE_OBJECT_DEPTH * 1000    # in mm
    
    for s in splits:
        rgb_dir = Path(f"./data/{scene_name}/{s}/")
        d_dir = Path(f"./data/{scene_name}/{s}_depth/")
        rgb_dir.mkdir(parents=True, exist_ok=True)
        d_dir.mkdir(exist_ok=True)
        transforms = {
            "camera_angle_x": 2 * np.arctan(680/(2*f_x)),        # AOV in x dimension; height and width are fixed in DeepDeform
            "camera_angle_y": 2 * np.arctan(480/(2*f_y)),        # AOV in y dimension
            "SCENE_OBJECT_DEPTH_at_extraction": SCENE_OBJECT_DEPTH,
            "frames": []
        }
        for i, frame in enumerate(splits[s]):
            # Copy RGB and depth files
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            shutil.copyfile(frame["rgb"],  f"{rgb_dir}/rgb_{num_str}.jpg")
            shutil.copyfile(frame["d"], f"{d_dir}/d_{num_str}.png")
            # Add transform and time info
            frame_info = {
                "file_path": f"./{s}/rgb_{num_str}",                # without .jpg
                "depth_file_path": f"./{s}_depth/d_{num_str}",      # without .png
                "rotation": 0,
                "time": frame["t"],
                "transform_matrix": transform_matrix.tolist(),        # Use the indentity for now
            }
            transforms["frames"].append(frame_info)
        
        with open(os.path.join(f"./data/{scene_name}", f"transforms_{s}.json"), "w", encoding='utf-8') as f:
            json.dump(transforms, f, ensure_ascii=False, indent=4)


def load_deepdeform_data(basedir, half_res=False, testskip=1, render_pose_type="spherical"):
    """Returns extracted DeepDeform data in compatible format. All output lengths are in meters.

    Args:
        basedir (str): Path to dataset.
        half_res (bool, optional): Whether to load img at half resolution. Defaults to False.
        testskip (int, optional): Use only every testskip-nd image. Defaults to 1.
        render_pose_type (str, optional): How to compute render poses. Defaults to "spherical".

    Returns:
        imgs: TODO
        TODO
    """
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            # load the per-frame file-locations, times and homogeneous transforms
            metas[s] = json.load(fp)

    all_imgs = []
    all_depth_maps = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]

        imgs = []
        depth_maps = []
        poses = []
        times = []
        # if s=='train' or testskip==0:
        #     skip = 2  # if you remove/change this 2, also change the /2 in the times vector
        # else:
        skip = testskip
            
        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.jpg')
            imgs.append(imageio.imread(fname))
            depth_maps.append(imageio.imread(os.path.join(basedir, frame['depth_file_path'] + '.png')))
            poses.append(np.array(frame['transform_matrix']))
            # if times are not given, assume frames are distributed uniformly over time
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
            times.append(cur_time)

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # .jpg has 3 channels -> RGB
        depth_maps = (np.array(depth_maps)).astype(np.float32)  
        depth_maps /= depth_maps.max() * 0.5       # convert depth from mm to [0, 2]
        poses = (np.array(poses)).astype(np.float32)
        poses[:, 0:3, 3] = poses[:, 0:3, 3] / (SCENE_OBJECT_DEPTH*1000)    # convert x,y,z from mm to [-1,1]
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_depth_maps.append(depth_maps)
        all_poses.append(poses)
        all_times.append(times)
    
    # prepare indices of train/val/test splits
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    depth_maps = np.concatenate(all_depth_maps, 0).reshape(-1, *depth_maps.shape[1:], 1)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    # relationship between the AOV and focal length: https://en.wikipedia.org/wiki/Angle_of_view
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])      # horizontal AOV
    camera_angle_y = float(meta['camera_angle_y'])      # horizontal AOV
    focal_x = .5 * W / np.tan(.5 * camera_angle_x)      # focal length
    focal_y = .5 * H / np.tan(.5 * camera_angle_y)

    # set the (novel) poses that are used to render novel views. Take them from the file, if given, else compute them from a sphere.
    if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format('render'))):
        with open(os.path.join(basedir, 'transforms_{}.json'.format('render')), 'r') as fp:
            meta = json.load(fp)
        render_poses = []
        for frame in meta['frames']:
            render_poses.append(np.array(frame['transform_matrix']))
        render_poses = np.array(render_poses).astype(np.float32)

    else:
        if render_pose_type == "spherical":
            render_poses = torch.stack([pose_spherical2(0, angle, 1) for angle in np.linspace(-20,20,120+1)], 0)       # changed from (-180,180,40+1)
        elif render_pose_type == "spiral": 
            render_poses = torch.stack([pose_spiral(angle, z_cam_dist, 1, H, W) for angle, z_cam_dist in              
                                        zip(np.linspace(0, 2*360, 120), np.linspace(0, -0.5, 120))], 0)

    render_times = torch.linspace(0., 1., render_poses.shape[0])
    
    if half_res:
        H = H//2
        W = W//2
        focal_x = focal_x/2.
        focal_y = focal_y/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        depth_maps_half_res = np.zeros((depth_maps.shape[0], H, W, 1))
        for i, (img, depth_map) in enumerate(zip(imgs, depth_maps)):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            depth_maps_half_res[i] = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_AREA).reshape(H, W, 1)
        imgs = imgs_half_res
        depth_maps = depth_maps_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, depth_maps, poses, times, render_poses, render_times, [H, W, focal_x, focal_y], i_split


if __name__ == "__main__":
    print("EXTRACTING DATA")
    # extract_deepdeform_data("/mnt/raid/kirwul/deepdeform/train/seq120", "office")
    # extract_deepdeform_data("/mnt/raid/kirwul/deepdeform/train/seq045", "human", end_frame_i=200)
    # extract_deepdeform_data("/mnt/raid/kirwul/deepdeform/train/seq150", "bag")
    #exit(0)

    # print("DEBUGGING")
    images, depth_maps, poses, times, render_poses, render_times, hwff, i_split = load_deepdeform_data("./data/human", True, 1)
    print('Loaded deepdeform', images.shape, render_poses.shape, hwff)