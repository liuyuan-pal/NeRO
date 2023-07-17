import argparse
from pathlib import Path

import numpy as np
import open3d

from eval_synthetic_shape import nearest_dist

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr',type=str,)
    parser.add_argument('--gt',type=str,)
    args = parser.parse_args()

    mesh_pr = open3d.io.read_triangle_mesh(f'{args.pr}')
    pts_pr = np.asarray(mesh_pr.vertices)
    mesh_gt = open3d.io.read_triangle_mesh(f'{args.gt}')
    pts_gt = np.asarray(mesh_gt.vertices)

    bn = 512
    dist_gt = nearest_dist(pts_gt, pts_pr, bn)
    dist_pr = nearest_dist(pts_pr, pts_gt, bn)

    stem = Path(args.pr).stem
    chamfer = (np.mean(dist_gt) + np.mean(dist_pr)) / 2
    results = f'{stem} {chamfer:.5f}'
    print(results)
    with open('data/geometry.log','a') as f:
        f.write(results+'\n')
