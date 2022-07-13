# DGD-NeRF: Depth-Guided Neural Radiance Field for Dynamic Scenes
### [[Website]](philippwulff.github.io/dgd-nerf/) [[Paper]](https://github.com/philippwulff/DGD-NeRF/blob/main/docs/Dynamic_NeRF_on_RGB_D_Data.pdf) 

DGD-NeRF is a method for synthesizing novel views, at an arbitrary point in time, of dynamic scenes with complex non-rigid geometries. We optimize an underlying deformable volumetric function from a sparse set of input monocular views without the need of ground-truth geometry nor multi-view images.

This project is an extension of [D-NeRF](https://github.com/albertpumarola/D-NeRF) improving modelling of dynamic scenes. We thank the authors of [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), [Dense Depth Priors for NeRF](https://github.com/barbararoessle/dense_depth_priors_nerf) and [Non-Rigid NeRF](https://github.com/facebookresearch/nonrigid_nerf) from whom be borrow code. 

![D-NeRF](docs/static/images/model.png)

## Installation
```
git clone https://github.com/philippwulff/D-NeRF.git
cd D-NeRF
conda create -n dgdnerf python=3.7
conda activate dgdnerf
pip install -r requirements.txt
```

If you want to directly explore the models or use our training data, you can download pre-trained models and the data:

**Download Pre-trained Weights**. You can download the pre-trained models as logs.zip from [here](https://github.com/philippwulff/DGD-NeRF/releases/tag/v1.0). Unzip the downloaded data to the project root dir in order to test it later. This is what the directory structure looks like:
```
├── logs 
│   ├── human
│   ├── bottle 
│   ├── gobblet 
│   ├── ...
```

## Dataset

**DeepDeform**. This is a RGB-D dataset of dynamic scenes with fixed camera poses. You can request access on the project's [GitHub page](https://github.com/AljazBozic/DeepDeform).

**Own Data**. Download our own dataset as data.zip from [here](https://github.com/philippwulff/DGD-NeRF/releases/tag/v1.0).
This is what the directory structure looks like with pretrained weights and dataset:
```
├── data 
│   ├── human
│   ├── bottle 
│   ├── gobblet 
│   ├── ...
├── logs 
│   ├── human
│   ├── bottle 
│   ├── gobblet 
│   ├── ...
```

**Generate Own scenes** Own scenes can be easily generated and integrated. We used an iPAD with Lidar Sensor (App: Record3d --> export Videos as EXR + RGB). Extract dataset to correct format by running load_owndataset.py (specifiy correct args in main() and create a scene configuration entry).

## How to Use It

If you have downloaded our data and the pre-trained weights, you can test our models without training them. Otherwise, you can also train a model with our or your own data from scratch.

### Demo
You can use these jupyter notebooks to explore the model.

| Description      | Jupyter Notebook |
| ----------- | ----------- |
| Synthesize novel views at an arbitrary point in time. (Requires trained model) | render.ipynb|
| Reconstruct the mesh at an arbitrary point in time. (Requires trained model) | reconstruct.ipynb|
| See the camera trajectory the training frames. | eda_owndataset_train.ipynb|
| See the camera poses of novel views. | eda_virtual_camera.ipynb|
| Visualize the sampling along camera rays. (Requires training logs) | eda_ray_sampling.ipynb|

The followinf instructions use the `human` scene as an example which can be replaced by the other scenes.

### Train
First download the dataset. Then,
```
conda activate dgdnerf
export PYTHONPATH='path/to/DGD-NeRF'
export CUDA_VISIBLE_DEVICES=0
python run_dnerf.py --config configs/human.txt
```

This command will run the `human` experiment with the specified args in the config `human.txt`.
Our extensions can be modularly enabled or disabled in `human.txt`.

### Test
First train the model or download pre-trained weights and dataset. Then, 
```
python run_dnerf.py --config configs/human.txt --render_only --render_test
```
This command will render the test set images of the `human` experiment. When finished, quantitative (`metrics.txt`) and qualitative (rgb and depth images/videos) results are saved to `./logs/human/renderonly_test_400000`.

### Render novel view videos

To render novel view images you can use the notebook `render.ipynb`. To render novel view videos run
```
python run_dnerf.py --config configs/human.txt --render_only --render_pose_type spherical
```

There exist multiple options for the render_pose_type dependent on the selected scene.

