# DGD-NeRF: Depth-Guided Neural Radiance Fields for Dynamic Scenes
### [[Website]](ourwebsite) [[Paper]](ourpaper) 

DGD-NeRF is a method for synthesizing novel views, at an arbitrary point in time, of dynamic scenes with complex non-rigid geometries. We optimize an underlying deformable volumetric function from a sparse set of input monocular views without the need of ground-truth geometry nor multi-view images.

This project is an extension of [D-NeRF](https://github.com/albertpumarola/D-NeRF) improving modelling of dynamic scenes. We thank the authors of [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), [Dense Depth Priors for NeRF](https://github.com/barbararoessle/dense_depth_priors_nerf) and [Non-Rigid NeRF](https://github.com/facebookresearch/nonrigid_nerf) from whom be borrow code. 

![D-NeRF](https://www.albertpumarola.com/images/2021/D-NeRF/model.png)

## Installation
```
git clone https://github.com/philippwulff/D-NeRF.git
cd D-NeRF
conda create -n dnerf python=3.6
conda activate dnerf
pip install -r requirements.txt
```

### Download Pre-trained Weights
 You can download the pre-trained models from [drive](https://drive.google.com/file/d/1VN-_DkRLL1khDVScQJEaohpbA2gC2I2K/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/25sveotbx2x7wap/logs.zip?dl=0). Unzip the downloaded data to the project root dir in order to test it later. See the following directory structure for an example:
```
├── logs 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

### Download Datasets

**DeepDeform**. This is a RGB-D dataset of dynamic scenes with fixed camera poses. You can request access on the project's [GitHub page](https://github.com/AljazBozic/DeepDeform).

**Own Data** TODO

## Usage
### Demo
We provide simple jupyter notebooks to explore the model. To use them first download the pre-trained weights and dataset.

| Description      | Jupyter Notebook |
| ----------- | ----------- |
| Synthesize novel views at an arbitrary point in time. | render.ipynb|
| Reconstruct mesh at an arbitrary point in time. | reconstruct.ipynb|
| Quantitatively evaluate trained model. | metrics.ipynb|

### Test
First download pre-trained weights and dataset. Then, 
```
python run_dnerf.py --config configs/mutant.txt --render_only --render_test
```
This command will run the `mutant` experiment. When finished, results are saved to `./logs/mutant/renderonly_test_799999` To quantitatively evaluate model run `metrics.ipynb` notebook

### Train
First download the dataset. Then,
```
conda activate dnerf
export PYTHONPATH='path/to/D-NeRF'
export CUDA_VISIBLE_DEVICES=0
python run_dnerf.py --config configs/mutant.txt
```

## Citation
If you use this code or ideas from the paper for your research, please cite our paper and the works we rely on:
  
```
TODO

@article{pumarola2020d,
  title={D-NeRF: Neural Radiance Fields for Dynamic Scenes},
  author={Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2011.13961},
  year={2020}
}
```