# NeRO for Blender Dataset

NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images
![](assets/teaser.jpg)

## [Project page](https://liuyuan-pal.github.io/NeRO/) | [Paper](https://arxiv.org/abs/2305.17398)

## Usage

### Setup

1. Install basic required packages.

```shell
git clone https://github.com/liuyuan-pal/NeRO.git
cd NeRO
pip install -r requirements.txt
```

2. Install `nvdiffrast`. Please follow instructions here [https://nvlabs.github.io/nvdiffrast/#installation](https://nvlabs.github.io/nvdiffrast/#installation).
3. Install `raytracing`. Please follow instructions here [https://github.com/ashawkey/raytracing](https://github.com/ashawkey/raytracing).

### Download datasets

- NeRO Models and datasets all can be found [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EvNz_o6SuE1MsXeVyB0VoQ0B9zL8NZXjQQg0KknIh6RKjQ?e=MaonKe).
- The **Blender** dataset used for testing comes from [NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and [Ref-NeRF](https://storage.googleapis.com/gresearch/refraw360/ref.zip).

### Config Files

This project is also compatible with the NeRO dataset. In this project, we have placed the setting of `dataset_dir` in the `.yaml` file. 

> For example, if the path of the lego dataset is `~/NeRO/data/nerf_synthetic/lego`, then in the `.yaml` file, `dataset_dir: ~/NeRO/data/nerf_synthetic`.

To run Blender Dataset, we need to change the `.yaml` files. The `.yaml` file for the Blender dataset differs from NeRO in **two ways**. 

- The category of database_name is nerf. 
- It is necessary to add `is_nerf: true` in the `.yaml` file.

Examples of the `.yaml` files can be seen in `configs/shape/nerf` and `configs/material/nerf`.

### Stage I: Shape reconstruction

1. In the `NeRO` directory, ensure that you have the following data:

```
NeRO
|-- data
    |-- GlossyReal
        |-- bear 
            ...
    |-- GlossySynthetic
        |-- bell
        
    |-- Blender
    	|-- lego
            ...
```

2. Run the training script

```shell
# reconstructing the "bell" of the Glossy Synthetic dataset
python run_training.py --cfg configs/shape/syn/bell.yaml

# reconstructing the "bear" of the Glossy Real dataset
python run_training.py --cfg configs/shape/real/bear.yaml

# reconstructing the "lego" of the Blender dataset
python run_training.py --cfg configs/shape/nerf/lego.yaml
```

Intermediate results will be saved at `data/train_vis`. Models will be saved at `data/model`.

3. Extract mesh from the model.

```shell
python extract_mesh.py --cfg configs/shape/syn/bell.yaml
python extract_mesh.py --cfg configs/shape/real/bear.yaml
python extract_mesh.py --cfg configs/shape/nerf/lego.yaml
```

The extracted meshes will be saved at `data/meshes`.

### Stage II: Material estimation

1. In the `NeRO` directory, ensure that you have the following data:

```
NeRO
|-- data
    |-- GlossyReal
        |-- bear 
            ...
    |-- GlossySynthetic
        |-- bell
        
    |-- Blender
    	|-- lego
            ...
    |-- meshes
        | -- bell_shape-300000.ply
        | -- bear_shape-300000.ply
        | -- lego_shape-300000.ply
             ...
```

2. Run the training script:

```shell
# estimate BRDF of the "bell" of the Glossy Synthetic dataset
python run_training.py --cfg configs/material/syn/bell.yaml

# estimate BRDF of the "bear" of the Glossy Real dataset
python run_training.py --cfg configs/material/real/bear.yaml

# estimate BRDF of the "lego" of the Blender dataset
python run_training.py --cfg configs/material/nerf/lego.yaml
```

Intermediate results will be saved at `data/train_vis`. Models will be saved at `data/model`.

3. Extract materials from the model.

```shell
python extract_materials.py --cfg configs/material/syn/bell.yaml
python extract_materials.py --cfg configs/material/real/bear.yaml
python extract_materials.py --cfg configs/material/nerf/lego.yaml
```

The extracted materials will be saved at `data/materials`.

### Relighting (Not Testing)

1. In the `NeRO` directory, ensure that you have the following data:

```
NeRO
|-- data
    |-- GlossyReal
        |-- bear 
            ...
    |-- GlossySynthetic
        |-- bell
            ...
    |-- meshes
        | -- bell_shape-300000.ply
        | -- bear_shape-300000.ply
             ...
    |-- materials
        | -- bell_material-100000
            | -- albedo.npy
            | -- metallic.npy
            | -- roughness.npy
        | -- bear_material-100000
            | -- albedo.npy
            | -- metallic.npy
            | -- roughness.npy
    |-- hdr
        | -- neon_photostudio_4k.exr
```

2. Run relighting script

```shell
python relight.py --blender <path-to-your-blender> \
                  --name bell-neon \
                  --mesh data/meshes/bell_shape-300000.ply \
                  --material data/materials/bell_material-100000 \
                  --hdr data/hdr/neon_photostudio_4k.exr \
                  --trans
                  
python relight.py --blender <path-to-your-blender> \
                  --name bear-neon \
                  --mesh data/meshes/bear_shape-300000.ply \
                  --material data/materials/bear_material-100000 \
                  --hdr data/hdr/neon_photostudio_4k.exr
```

The relighting results will be saved at `data/relight` with the directory name of `bell-neon` or `bear-neon`. This command means that we use `neon_photostudio_4k.exr` to relight the object.


### Training on custom objects

Refer to [custom_object.md](custom_object.md).

### Evaluation

Refer to [eval.md](eval.md).



## Results (on Blender Datasets)

### Stage1: Shape reconstruction

| NeRF      | PSNR     | Ref-NeRF | PSNR     |
| --------- | -------- | -------- | -------- |
| chair     | 27.73918 | ball     | 39.62966 |
| drums     | 21.06286 | car      | 26.09878 |
| ficus     | 22.51317 | coffee   | 30.61401 |
| hotdog    | 29.33451 | helmet   | 29.56573 |
| lego      | 23.47746 | teapot   | 35.41234 |
| materials | 24.32323 | toaster  | 25.23647 |
| mic       | 24.54512 |          |          |
| ship      | 22.91336 |          |          |

### Stage2: Material estimation

| NeRF      | PSNR     | Ref-NeRF | PSNR     |
| --------- | -------- | -------- | -------- |
| chair     | 28.74847 | ball     | 33.66338 |
| drums     | 24.88227 | car      | 26.9762  |
| ficus     | 28.38085 | coffee   | 33.76237 |
| hotdog    | 32.13475 | helmet   | 29.59044 |
| lego      | 25.66081 | teapot   | 40.28731 |
| materials | 24.8514  | toaster  | 27.30664 |
| mic       | 28.63963 |          |          |
| ship      | 26.54597 |          |          |

## Acknowledgements

In this repository, we have used codes from the following repositories. 
We thank all the authors for sharing great codes.

- [NeuS](https://github.com/Totoro97/NeuS)
- [NvDiffRast](https://github.com/NVlabs/nvdiffrast)
- [NvDiffRec](https://github.com/NVlabs/nvdiffrec)
- [Ref-NeRF](https://github.com/google-research/multinerf)
- [RayTracing](https://github.com/ashawkey/raytracing)
- [COLMAP](https://colmap.github.io/)

## Citation

```
@inproceedings{liu2023nero,
  title={NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images},
  author={Liu, Yuan and Wang, Peng and Lin, Cheng and Long, Xiaoxiao and Wang, Jiepeng and Liu, Lingjie and Komura, Taku and Wang, Wenping},
  booktitle={SIGGRAPH},
  year={2023}
}
```
