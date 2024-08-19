#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

def check_and_create_env(env_name, python_version, requirements=None):
    result = subprocess.run(f"conda env list | grep {env_name}", shell=True, capture_output=True, text=True)
    if env_name in result.stdout:
        print(f"{env_name} environment already exists.")
    else:
        print(f"Creating {env_name} environment...")
        run_command(f"conda create -n {env_name} python={python_version} -y")
        
        if env_name == "colmap_env":
            run_command(f"conda activate {env_name} && conda install -c conda-forge colmap -y")
        elif env_name == "nero_env" and requirements:
            run_command(f"conda activate {env_name} && pip install -r {requirements}")

def load_colmap():
    run_command("module load libs/nvidia-hpc-sdk/23.9")
    check_and_create_env("colmap_env", "3.10")
    run_command("source activate colmap_env")
    print("COLMAP environments loaded!")

def load_nero(nero_dir):
    run_command("module load libs/nvidia-hpc-sdk/22.3")
    run_command("module load apps/binapps/anaconda3/2022.10")
    run_command("module load libs/cuda/11.7.0")
    requirements_file = Path(nero_dir) / "requirements.txt"
    check_and_create_env("nero_env", "3.10", requirements_file)
    run_command("source activate nero_env")
    print("NERO env loaded!")

def create_config_file(config_dir, dataset_name, config_type):
    config_file = config_dir / f"{dataset_name}_{config_type}.yaml"
    config = {
        "name": f"{dataset_name}_{config_type}",
        "network": config_type,
        "database_name": f"custom/{dataset_name}/raw_2000",
    }

    if config_type == "shape":
        config.update({
            "shader_config": {
                "human_light": True
            },
            "apply_occ_loss": True,
            "occ_loss_step": 20000,
            "clip_sample_variance": False,
            "loss": ['nerf_render', 'eikonal', 'std', 'init_sdf_reg', 'occ'],
            "val_metric": ['shape_render'],
            "key_metric_name": "psnr",
            "eikonal_weight": 0.1,
            "freeze_inv_s_step": 15000,
            "train_dataset_type": "dummy",
            "train_dataset_cfg": {
                "database_name": f"custom/{dataset_name}/raw_2000"
            },
            "val_set_list": [{
                "name": "val",
                "type": "dummy",
                "cfg": {
                    "database_name": f"custom/{dataset_name}/raw_2000"
                }
            }],
            "optimizer_type": "adam",
            "lr_type": "warm_up_cos",
            "lr_cfg": {},
            "total_step": 300000,
            "val_interval": 5000,
            "save_interval": 1000,
            "train_log_step": 20
        })
    elif config_type == "material":
        config.update({
            "mesh": f"data/meshes/{dataset_name}_shape-300000.ply",
            "reg_diffuse_light": True,
            "reg_diffuse_light_lambda": 0.1,
            "reg_mat": True,
            "shader_cfg": {
                "diffuse_sample_num": 512,
                "specular_sample_num": 256,
                "outer_light_version": "sphere_direction",
                "light_exp_max": 5.0,
                "inner_light_exp_max": 5.0,
                "human_lights": True
            },
            "loss": ['nerf_render', 'mat_reg'],
            "val_metric": ['mat_render'],
            "key_metric_name": "psnr",
            "train_dataset_type": "dummy",
            "train_dataset_cfg": {
                "database_name": f"custom/{dataset_name}/raw_2000"
            },
            "val_set_list": [{
                "name": "val",
                "type": "dummy",
                "cfg": {
                    "database_name": f"custom/{dataset_name}/raw_2000"
                }
            }],
            "optimizer_type": "adam",
            "lr_type": "warm_up_cos",
            "lr_cfg": {
                "end_warm": 1000,
                "end_iter": 100000
            },
            "total_step": 100000,
            "val_interval": 5000,
            "save_interval": 500,
            "train_log_step": 10
        })

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def reorganize_directory(project_dir):
    # Create colmap directory if it doesn't exist
    colmap_dir = project_dir / "colmap"
    colmap_dir.mkdir(exist_ok=True)

    # Move everything except object_point_cloud.ply, meta_info.txt, and images directory to colmap directory
    for item in project_dir.iterdir():
        if item.name not in ["object_point_cloud.ply", "meta_info.txt", "images", "colmap"]:
            shutil.move(str(item), str(colmap_dir))

    print("Directory structure reorganized.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python nero_pipeline.py <path_to_image_folder> <path_to_nero_directory>")
        sys.exit(1)

    image_folder = Path(sys.argv[1])
    nero_dir = Path(sys.argv[2])

    if not image_folder.is_dir():
        print(f"Error: {image_folder} is not a valid directory")
        sys.exit(1)

    if not nero_dir.is_dir():
        print(f"Error: {nero_dir} is not a valid directory")
        sys.exit(1)

    dataset_name = image_folder.name
    project_dir = Path("data/custom") / dataset_name
    config_dir = Path("configs/custom")

    # Step 1: Setup COLMAP environment
    print("Step 1: Setting up COLMAP environment")
    load_colmap()

    # Step 2: Prepare dataset
    print("Step 2: Preparing dataset")
    project_dir.mkdir(parents=True, exist_ok=True)
    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for img in image_folder.glob("*"):
        shutil.copy(img, images_dir)

    # Step 3: Run COLMAP
    print("Step 3: Running COLMAP")
    run_command(f"python run_colmap.py --project_dir {project_dir} --same_camera")

    # Step 4: Reorganize directory
    print("Step 4: Reorganizing directory structure")
    reorganize_directory(project_dir)

    # Step 5: Process point cloud (automated)
    print("Step 5: Processing point cloud")
    # Here you would typically add code to automatically process the point cloud
    # For this example, we'll create a dummy meta_info.txt file if it doesn't exist
    meta_info_file = project_dir / "meta_info.txt"
    if not meta_info_file.exists():
        with open(meta_info_file, "w") as f:
            f.write("plane_primitive: 0.0551935 0.826918 -0.559607\n")
            f.write("distance: -0.077443 -0.00604 -0.014601\n")

    # Step 6: NerO environment setup
    print("Step 6: Setting up NerO environment")
    load_nero(nero_dir)

    # Step 7: Create config files
    print("Step 7: Creating configuration files")
    config_dir.mkdir(parents=True, exist_ok=True)
    create_config_file(config_dir, dataset_name, "shape")
    create_config_file(config_dir, dataset_name, "material")

    # Step 8: Shape training and extraction
    print("Step 8: Training and extracting shape")
    run_command(f"python run_training.py --cfg {config_dir}/{dataset_name}_shape.yaml")
    run_command(f"python extract_mesh.py --cfg {config_dir}/{dataset_name}_shape.yaml")

    # Step 9: Material training and extraction
    print("Step 9: Training and extracting material")
    run_command(f"python run_training.py --cfg {config_dir}/{dataset_name}_material.yaml")
    run_command(f"python extract_materials.py --cfg {config_dir}/{dataset_name}_material.yaml")

    # Step 10: Colored mesh extraction
    print("Step 10: Extracting colored mesh")
    run_command(f"python extract_colored_mesh_high_res.py --cfg {config_dir}/{dataset_name}_material.yaml")

    print("NerO Pipeline completed successfully!")

if __name__ == "__main__":
    main()