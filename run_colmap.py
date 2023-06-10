import argparse
import subprocess
import os
from pathlib import Path

import numpy as np
from skimage.io import imread

from colmap.database import COLMAPDatabase
from colmap.read_write_model import CAMERA_MODEL_NAMES

def run_sfm(image_dir, project_dir, same_camera=False, colmap_path='$HOME/code/colmap/build/src/exe/colmap'):
    Path(project_dir).mkdir(exist_ok=True, parents=True)
    # create database for all images
    db = COLMAPDatabase.connect(f'{project_dir}/database.db')
    db.create_tables()

    # add images
    img_dir = Path(image_dir)
    img_fns = []
    for pattern in ['*.jpg','*.png','*.PNG','*.JPG']:
        img_fns+=[fn for fn in img_dir.glob(pattern)]
    img_fns = sorted(img_fns)
    global_cam_id = None
    for k, img_fn in enumerate(img_fns):
        img = imread(img_fn)
        h, w, _ = img.shape
        focal = np.sqrt(h**2+w**2) # guess a focal here
        if same_camera:
            if k==0: global_cam_id = db.add_camera(CAMERA_MODEL_NAMES['SIMPLE_PINHOLE'].model_id,
                                                   float(w), float(h), np.array([focal, w/2, h/2], np.float64), prior_focal_length=True)
            db.add_image(img_fn.name, global_cam_id)
        else:
            cam_id = db.add_camera(CAMERA_MODEL_NAMES['SIMPLE_PINHOLE'].model_id,
                                   float(w), float(h), np.array([focal, w/2,h/2],np.float64),prior_focal_length=True)
            db.add_image(img_fn.name, cam_id)

    db.commit()
    db.close()

    # feature extraction
    cmd=[colmap_path,'feature_extractor',
         '--database_path',f'{project_dir}/database.db',
         '--image_path',f'{image_dir}']
    print(' '.join(cmd))
    subprocess.run(cmd,check=True)

    # feature matching
    cmd=[colmap_path,'exhaustive_matcher',
         '--database_path',f'{project_dir}/database.db']
    print(' '.join(cmd))
    subprocess.run(cmd,check=True)

    # SfM
    Path(f'{project_dir}/sparse').mkdir(exist_ok=True,parents=True)
    cmd=[colmap_path,'mapper',
         '--database_path',f'{project_dir}/database.db',
         '--image_path',f'{image_dir}',
         '--output_path',f'{project_dir}/sparse']
    print(' '.join(cmd))
    subprocess.run(cmd,check=True)

    # dense reconstruction
    Path(f'{project_dir}/dense').mkdir(exist_ok=True,parents=True)
    cmd=[colmap_path,'image_undistorter',
         '--image_path',f'{image_dir}',
         '--input_path',f'{project_dir}/sparse/0',
         '--output_path',f'{project_dir}/dense']
    print(' '.join(cmd))
    subprocess.run(cmd,check=True)

    cmd=[colmap_path,'patch_match_stereo',
         '--workspace_path',f'{project_dir}/dense']
    print(' '.join(cmd))
    subprocess.run(cmd,check=True)

    cmd = [str(colmap_path), 'stereo_fusion',
           '--workspace_path', f'{project_dir}/dense',
           '--workspace_format', 'COLMAP',
           '--input_type', 'geometric',
           '--output_path', f'{project_dir}/points.ply',
           # '--StereoFusion.num_threads','14',
           '--StereoFusion.check_num_images','5',
           # '--StereoFusion.cache_size','14',
           # '--StereoFusion.use_cache','1',
           # '--StereoFusion.max_image_size','1024'
           ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str)
    parser.add_argument('--colmap', type=str)
    parser.add_argument('--same_camera', action='store_true', default=False, dest='same_camera')
    args = parser.parse_args()
    image_dir = f'{args.project_dir}/images'
    run_sfm(image_dir, args.project_dir, args.same_camera, colmap_path=args.colmap)

if __name__=="__main__":
    main()
