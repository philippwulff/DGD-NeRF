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
import OpenEXR as exr
import Imath

from utils.load_blender import trans_t, rot_phi, rot_theta
#from load_blender import trans_t, rot_phi, rot_theta

#TODO: Set render_poses correct, maybe even test_poses

SCENE_OBJECT_DEPTH = 1.45         # Distance to the main object of the scene in meters
FPS = 15

def quat_and_trans_2_trans_matrix(Q):
    """
    Covert a quaternion plus transformation vector into a full four-dimensional translation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion and translation vector (q0,q1,q2,q3,tx,ty,tz) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    tx = Q[4]
    ty = Q[5]
    tz = Q[6]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    trans_matrix = np.array([[r00, r01, r02, tx],
                           [r10, r11, r12, ty],
                           [r20, r21, r22, tz],
                           [0,0,0,1]])
                            
    return trans_matrix

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
    return c2w#, (x_w, y_w, z_w)

def pose_static(i,pose):
    """Returns static camera pose of selected pose

    Args:
        i (int): counter for frame index
        pose: shape 4x4
    """
    return pose

def poses_original_trajectory():
    """Returns poses from not used frames of the original video
    """
    with open("data/EXR_RGBD" + "/metadata.json", "r") as f:
        metas = json.load(f)
        
        poses = np.array(metas["poses"]).astype(np.float32)
        poses = poses[2::int(60/FPS)]

        transform_matrix = np.zeros(shape=(len(poses),4,4))
        R_z = np.array([[np.cos(np.pi), -np.sin(np.pi),0],[np.sin(np.pi), np.cos(np.pi), 0],[0,0,1]]).astype(np.float32)
        for i in range(len(poses)):
            transform_matrix[i] = quat_and_trans_2_trans_matrix(poses[i])
            transform_matrix[i,2,3] += SCENE_OBJECT_DEPTH
            transform_matrix[i,:3,:3] = transform_matrix[i,:3,:3] @ R_z # Rotate every transform_matrix 180 degree around z-axis
        return transform_matrix


def extract_owndataset_data(datadir, scene_name, start_frame_i=0, end_frame_i=None, step=1, train_p=0.7, val_p=0.15, test_p=0.15):
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
    def get_number_only(elem):
        if len(elem)==5:
            i = elem[:1]
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            return num_str
        if len(elem)==6:
            i = elem[:2]
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            return num_str
        if len(elem)==7:
            i = elem[:3]
            num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
            return num_str

    rgb_paths = sorted(os.listdir(os.path.join(datadir, "color")), key=get_number_only)[start_frame_i:end_frame_i:step]
    depth_paths = sorted(os.listdir(os.path.join(datadir, "depth")), key=get_number_only)[start_frame_i:end_frame_i:step]
    rgb_paths = rgb_paths[::int(60/FPS)]          # Control fps --> original 60fps, reduced to 15fps
    depth_paths = depth_paths[::int(60/FPS)]      # Control fps --> original 60fps, reduced to 15fps
    if not end_frame_i:
        end_frame_i = len(rgb_paths) - 1
    assert len(rgb_paths) == len(depth_paths), "Unequal number of RGB and depth frames."


    with open("data/EXR_RGBD" + "/metadata.json", "r") as f:
            metas = json.load(f)

            K_matrix = np.array(metas["K"]).astype(np.float32)
            f_x = K_matrix[0]
            f_y = K_matrix[4]

            H = np.array(metas["h"]).astype(np.float32)
            W = np.array(metas["w"]).astype(np.float32)

            #init_pose = np.array(metas["initPose"]).astype(np.float32)
            #transform_matrix = quat_and_trans_2_trans_matrix(init_pose) #J: i think it is in m
            #transform_matrix[2, 3] = SCENE_OBJECT_DEPTH #FIXME J: To set training cam higher
            
            poses = np.array(metas["poses"]).astype(np.float32)
            poses = poses[::int(60/FPS)]
            
            transform_matrix = np.zeros(shape=(len(poses),4,4))
            R_z = np.array([[np.cos(np.pi), -np.sin(np.pi),0],[np.sin(np.pi), np.cos(np.pi), 0],[0,0,1]]).astype(np.float32)
            
            for z in range(len(poses)):
                transform_matrix[z] = quat_and_trans_2_trans_matrix(poses[z])
                transform_matrix[z,2,3] += SCENE_OBJECT_DEPTH
                transform_matrix[z,:3,:3] = transform_matrix[z,:3,:3] @ R_z # Rotate every transform_matrix 180 degree around z-axis

    frames = [{"rgb": os.path.join(datadir, "color", rgb), "d": os.path.join(datadir, "depth", d), "t": t, "transform_matrix": transform_matrix} 
                    for rgb, d, t, transform_matrix in zip(rgb_paths, depth_paths, np.linspace(0, 1, len(rgb_paths)), transform_matrix)]
    np.random.shuffle(frames)

    # Create train-val-test splits from shuffled frames and then sort by time
    splits = {
        "train": sorted(frames[:int(len(frames)*train_p)], key=lambda x: x["t"]), 
        "val": sorted(frames[int(len(frames)*train_p):int(len(frames)*(train_p+val_p))], key=lambda x: x["t"]), 
        "test": sorted(frames[int(len(frames)*(train_p+val_p)):], key=lambda x: x["t"]),
    }
    print(f"Creating {int(train_p*100)}-{int(val_p*100)}-{int(test_p*100)}-Split with {len(splits['train'])}-{len(splits['val'])}-{len(splits['test'])} images.")


    for i,s in enumerate(splits):
            rgb_dir = Path(f"./data/{scene_name}/{s}/")
            d_dir = Path(f"./data/{scene_name}/{s}_depth/")
            rgb_dir.mkdir(parents=True, exist_ok=True)
            d_dir.mkdir(exist_ok=True)
            transforms = {
                "camera_angle_x": 2 * np.arctan(W/(2*f_x)),        # AOV in x dimension; height and width are fixed in DeepDeform
                "camera_angle_y": 2 * np.arctan(H/(2*f_y)),        # AOV in y dimension
                "SCENE_OBJECT_DEPTH_at_extraction": SCENE_OBJECT_DEPTH,
                "frames": []
            }
            for i, frame in enumerate(splits[s]):
                # Copy RGB and depth files
                num_str = "".join(["0" for _ in range(0, 3-len(str(i)))]) + str(i)
                shutil.copyfile(frame["rgb"],  f"{rgb_dir}/rgb_{num_str}.jpg")
                shutil.copyfile(frame["d"], f"{d_dir}/d_{num_str}.exr")
                # Add transform and time info
                frame_info = {
                    "file_path": f"./{s}/rgb_{num_str}",                # without .jpg
                    "depth_file_path": f"./{s}_depth/d_{num_str}",      # without .png
                    "rotation": 0,
                    "time": frame["t"],
                    "transform_matrix": frame["transform_matrix"].tolist(),        # Use the indentity for now
                }
                transforms["frames"].append(frame_info)
            
            with open(os.path.join(f"./data/{scene_name}", f"transforms_{s}.json"), "w", encoding='utf-8') as f:
                json.dump(transforms, f, ensure_ascii=False, indent=4)


def load_owndataset_data(basedir, half_res=False, testskip=1, render_pose_type="spherical", slowmo=False):
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
            depth_exr = exr.InputFile(os.path.join(basedir, frame['depth_file_path'] + '.exr'))
            header = depth_exr.header()
            dw = header['dataWindow']
            size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
            for c in header['channels']:
                depth_map = depth_exr.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
                depth_map = np.fromstring(depth_map, dtype=np.float32)
                depth_map = np.reshape(depth_map, size) 
            depth_maps.append(depth_map)
            poses.append(np.array(frame['transform_matrix']))
            # if times are not given, assume frames are distributed uniformly over time
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip])-1)
            times.append(cur_time)

        # DeepDeform does not have multiple views at t=0
        # assert times[0] == 0, "Time must start at 0"

        imgs = (np.array(imgs) / 255.).astype(np.float32)  # .jpg has 3 channels -> RGB
        depth_maps = (np.array(depth_maps)).astype(np.float32)  # convert mm to m
        poses = (np.array(poses)).astype(np.float32)

        scaling_factor = np.max(np.abs(poses[:,:3,3]))  #convert to have poses in unit cube [-1,1]^3
        depth_maps /= scaling_factor                    #convert to have same unit as in poses
        poses[:, 0:3, 3] /= scaling_factor

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
    focal_x = .5 * W / np.tan(.5 * camera_angle_x)        # focal length
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
            render_poses = torch.stack([pose_spherical2(0, angle, SCENE_OBJECT_DEPTH) for angle in np.linspace(0,90,101)], 0)       # changed from (-180,180,40+1)
            render_poses[:, 0:3, 3] /= scaling_factor # scale render_poses as poses        
        elif render_pose_type == "spiral": 
            render_poses = torch.stack([pose_spiral(angle, z_cam_dist, SCENE_OBJECT_DEPTH, H, W) for angle, z_cam_dist in              
                                        zip(np.linspace(0, 2*360, 101), np.linspace(0, -SCENE_OBJECT_DEPTH*0.5, 101))], 0)
            render_poses[:, 0:3, 3] /= scaling_factor # scale render_poses as poses
        elif render_pose_type == "static":
            render_poses = torch.stack([pose_static(i,torch.tensor(poses[0,:,:])) for i in range(101)],0) #FIXME J: not tested
        elif render_pose_type == "original_trajectory": 
            render_poses = torch.tensor(poses_original_trajectory(), dtype=torch.float32)
            render_poses[:, 0:3, 3] /= scaling_factor # scale render_poses as poses #FIXME J: Not tested

    render_times = torch.linspace(0., 1., render_poses.shape[0])
    if slowmo:
        render_times_0 = torch.linspace(0., 0.35, int(render_poses.shape[0]*0.35))
        render_times_2 = torch.linspace(0.5,0.85, int(render_poses.shape[0]*0.35))
        render_times_1 = torch.linspace(0.35,0.5, render_poses.shape[0]-2*int(render_poses.shape[0]*0.35)) #slowmo
        render_times = torch.cat((render_times_0, render_times_1))
        render_times = torch.cat((render_times, render_times_2))
        print(render_times)
    
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

    return imgs, depth_maps, poses, times, render_poses, render_times, [H, W, focal_x, focal_y], i_split #FIXME J: depth_maps/1000


if __name__ == "__main__":
    print("EXTRACTING DATA")
    extract_owndataset_data("data/EXR_RGBD", "johannes_2")
    #exit(0)

    # print("DEBUGGING")
    # images, depth_maps, poses, times, render_poses, render_times, hwf, i_split = load_deepdeform_data("./data/human", True, 1)
    # print('Loaded deepdeform', images.shape, render_poses.shape, hwf)