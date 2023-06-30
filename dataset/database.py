import abc
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread, imsave
from tqdm import tqdm

from colmap import plyfile
from colmap.read_write_model import read_model
from utils.base_utils import resize_img, read_pickle, project_points, save_pickle, pose_inverse, \
    mask_depth_to_pts, pose_apply
import open3d as o3d
import json
import imageio

from utils.pose_utils import look_at_crop


class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):  # gt poses
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    @abc.abstractmethod
    def get_depth(self, img_id):
        pass


def crop_by_points(img, ref_points, pose, K, size):
    h, w, _ = img.shape
    pts2d, depth = project_points(ref_points, pose, K)
    pts2d[:, 0] = np.clip(pts2d[:, 0], a_min=0, a_max=w - 1)
    pts2d[:, 1] = np.clip(pts2d[:, 1], a_min=0, a_max=h - 1)
    pt_min, pt_max = np.min(pts2d, 0), np.max(pts2d, 0)

    region_size = np.max(pt_max - pt_min)
    region_size = min(region_size, h - 3, w - 3)  # cannot exceeds image size

    x_size, y_size = pt_max - pt_min
    x_min, y_min = pt_min
    x_max, y_max = pt_max
    if region_size <= x_size:
        x_cen = (x_min + x_max) / 2
    elif region_size > x_size:
        b0 = max(region_size / 2, x_max - region_size / 2)
        b1 = min(x_min + region_size / 2, w - 2 - region_size / 2)
        x_cen = (b0 + b1) / 2
    if region_size <= y_size:
        y_cen = (y_min + y_max) / 2
    elif region_size > y_size:
        b0 = max(region_size / 2, y_max - region_size / 2)
        b1 = min(y_min + region_size / 2, h - 2 - region_size / 2)
        y_cen = (b0 + b1) / 2

    center = np.asarray([x_cen, y_cen], np.float32)
    scale = size / region_size
    img1, K1, pose1, pose_rect, H = look_at_crop(img, K, pose, center, 0, scale, size, size)
    return img1, K1, pose1


class GlossyRealDatabase(BaseDatabase):
    meta_info = {
        'bear': {'forward': np.asarray([0.539944, -0.342791, 0.341446], np.float32),
                 'up': np.asarray((0.0512875, -0.645326, -0.762183), np.float32), },
        'coral': {'forward': np.asarray([0.004226, -0.235523, 0.267582], np.float32),
                  'up': np.asarray((0.0477973, -0.748313, -0.661622), np.float32), },
        'maneki': {'forward': np.asarray([-2.336584, -0.406351, 0.482029], np.float32),
                   'up': np.asarray((-0.0117387, -0.738751, -0.673876), np.float32), },
        'bunny': {'forward': np.asarray([0.437076, -1.672467, 1.436961], np.float32),
                  'up': np.asarray((-0.0693234, -0.644819, -.761185), np.float32), },
        'vase': {'forward': np.asarray([-0.911907, -0.132777, 0.180063], np.float32),
                 'up': np.asarray((-0.01911, -0.738918, -0.673524), np.float32), },
    }

    def __init__(self, database_name, dataset_dir):
        super().__init__(database_name)
        _, self.object_name, self.max_len = database_name.split('/')

        self.root = f'{dataset_dir}/{self.object_name}'
        self._parse_colmap()
        self._normalize()
        if not self.max_len.startswith('raw'):
            self.max_len = int(self.max_len)
            self.image_dir = ''
            self._crop()
        else:
            h, w, _ = imread(f'{self.root}/images/{self.image_names[self.img_ids[0]]}').shape
            max_len = int(self.max_len.split('_')[1])
            ratio = float(max_len) / max(h, w)
            th, tw = int(ratio * h), int(ratio * w)
            rh, rw = th / h, tw / w

            Path(f'{self.root}/images_{self.max_len}').mkdir(exist_ok=True, parents=True)
            for img_id in tqdm(self.img_ids):
                if not Path(f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}').exists():
                    img = imread(f'{self.root}/images/{self.image_names[img_id]}')
                    img = resize_img(img, ratio)
                    imsave(f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}', img)

                K = self.Ks[img_id]
                self.Ks[img_id] = np.diag([rw, rh, 1.0]) @ K

    def _parse_colmap(self):
        if Path(f'{self.root}/cache.pkl').exists():
            self.poses, self.Ks, self.image_names, self.img_ids = read_pickle(f'{self.root}/cache.pkl')
        else:
            cameras, images, points3d = read_model(f'{self.root}/colmap/sparse/0')

            self.poses, self.Ks, self.image_names, self.img_ids = {}, {}, {}, []
            for img_id, image in images.items():
                self.img_ids.append(img_id)
                self.image_names[img_id] = image.name

                R = image.qvec2rotmat()
                t = image.tvec
                pose = np.concatenate([R, t[:, None]], 1).astype(np.float32)
                self.poses[img_id] = pose

                cam_id = image.camera_id
                camera = cameras[cam_id]
                if camera.model == 'SIMPLE_RADIAL':
                    f, cx, cy, _ = camera.params
                elif camera.model == 'SIMPLE_PINHOLE':
                    f, cx, cy = camera.params
                else:
                    raise NotImplementedError
                self.Ks[img_id] = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1], ], np.float32)

            save_pickle([self.poses, self.Ks, self.image_names, self.img_ids], f'{self.root}/cache.pkl')

    def _load_point_cloud(self, pcl_path):
        with open(pcl_path, "rb") as f:
            plydata = plyfile.PlyData.read(f)
            xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
        return xyz

    def _compute_rotation(self, vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert / np.linalg.norm(vert)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R

    def _normalize(self):
        ref_points = self._load_point_cloud(f'{self.root}/object_point_cloud.ply')
        max_pt, min_pt = np.max(ref_points, 0), np.min(ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        offset = -center  # x1 = x0 + offset
        scale = 1 / np.max(np.linalg.norm(ref_points - center[None, :], 2, 1))  # x2 = scale * x1
        up, forward = self.meta_info[self.object_name]['up'], self.meta_info[self.object_name]['forward']
        up, forward = up / np.linalg.norm(up), forward / np.linalg.norm(forward)
        R_rec = self._compute_rotation(up, forward)  # x3 = R_rec @ x2
        self.ref_points = scale * (ref_points + offset) @ R_rec.T
        self.scale_rect = scale
        self.offset_rect = offset
        self.R_rect = R_rec

        # x3 = R_rec @ (scale * (x0 + offset))
        # R_rec.T @ x3 / scale - offset = x0

        # pose [R,t] x_c = R @ x0 + t
        # pose [R,t] x_c = R @ (R_rec.T @ x3 / scale - offset) + t
        # x_c = R @ R_rec.T @ x3 + (t - R @ offset) * scale
        # R_new = R @ R_rec.T    t_new = (t - R @ offset) * scale
        for img_id, pose in self.poses.items():
            R, t = pose[:, :3], pose[:, 3]
            R_new = R @ R_rec.T
            t_new = (t - R @ offset) * scale
            self.poses[img_id] = np.concatenate([R_new, t_new[:, None]], -1)

    def _crop(self):
        if Path(f'{self.root}/images_{self.max_len}/meta_info.pkl').exists():
            self.poses, self.Ks = read_pickle(f'{self.root}/images_{self.max_len}/meta_info.pkl')
        else:
            poses_new, Ks_new = {}, {}
            print('cropping images ...')
            Path(f'{self.root}/images_{self.max_len}').mkdir(exist_ok=True, parents=True)
            for img_id in tqdm(self.img_ids):
                pose, K = self.poses[img_id], self.Ks[img_id]
                img = imread(f'{self.root}/images/{self.image_names[img_id]}')
                img1, K1, pose1 = crop_by_points(img, self.ref_points, pose, K, self.max_len)
                imsave(f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}', img1)
                poses_new[img_id] = pose1
                Ks_new[img_id] = K1

            save_pickle([poses_new, Ks_new], f'{self.root}/images_{self.max_len}/meta_info.pkl')
            self.poses, self.Ks = poses_new, Ks_new

    def get_image(self, img_id):
        img = imread(f'{self.root}/images_{self.max_len}/{self.image_names[img_id]}')
        return img

    def get_K(self, img_id):
        K = self.Ks[img_id]
        return K.copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    def get_mask(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        img = self.get_image(img_id)
        h, w, _ = img.shape
        return np.ones([h, w], np.float32), np.ones([h, w], np.bool)


class GlossySyntheticDatabase(BaseDatabase):
    def __init__(self, database_name, dataset_dir):
        super().__init__(database_name)
        _, model_name = database_name.split('/')
        RENDER_ROOT = dataset_dir
        self.root = f'{RENDER_ROOT}/{model_name}'
        self.img_num = len(glob.glob(f'{self.root}/*.pkl'))
        self.img_ids = [str(k) for k in range(self.img_num)]
        self.cams = [read_pickle(f'{self.root}/{k}-camera.pkl') for k in range(self.img_num)]
        self.scale_factor = 1.0

    def get_image(self, img_id):
        return imread(f'{self.root}/{img_id}.png')[..., :3]

    def get_K(self, img_id):
        K = self.cams[int(img_id)][1]
        return K.astype(np.float32)

    def get_pose(self, img_id):
        pose = self.cams[int(img_id)][0].copy()
        pose = pose.astype(np.float32)
        pose[:, 3:] *= self.scale_factor
        return pose

    def get_img_ids(self):
        return self.img_ids

    def get_depth(self, img_id):
        assert (self.scale_factor == 1.0)
        depth = imread(f'{self.root}/{img_id}-depth.png')
        depth = depth.astype(np.float32) / 65535 * 15
        mask = depth < 14.5
        return depth, mask

    def get_mask(self, img_id):
        raise NotImplementedError


class NeRFSyntheticDatabase(BaseDatabase):
    def __init__(self, database_name, dataset_dir, testskip=8):
        super().__init__(database_name)
        _, model_name = database_name.split('/')
        RENDER_ROOT = dataset_dir
        #RENDER_ROOT = '/media/data_nix/yzy/Git_Project/data/nerf_synthetic'
        self.root = f'{RENDER_ROOT}/{model_name}'
        self.scale_factor = 1.0

        splits = ['train', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(self.root, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(self.root, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(2)]

        self.imgs = np.concatenate(all_imgs, 0)
        self.poses = np.concatenate(all_poses, 0)
        self.poses[..., :3, 3] /= 2

        self.img_num = self.imgs.shape[0]
        self.img_ids = [str(k) for k in range(self.img_num)]

        H, W = self.imgs[0].shape[:2]

        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.Ks = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    def get_image(self, img_id):
        imgs = self.imgs[int(img_id)]
        return imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
        # return imread(f'{self.root}/{img_id}.png')[..., :3]

    def get_K(self, img_id):
        K = self.Ks
        return K.astype(np.float32)

    def get_pose(self, img_id):
        pose = self.poses[int(img_id)].copy()[:3, :]
        # pose = self.cams[int(img_id)][0].copy()
        pose = pose.astype(np.float32)
        pose[:, 3:] *= self.scale_factor
        return pose

    def get_img_ids(self):
        return self.img_ids

    def get_depth(self, img_id):
        assert (self.scale_factor == 1.0)
        depth = torch.randn(800, 800, 3).cpu().numpy()
        # depth = imread(f'{self.root}/test/r_{img_id}_depth_0001.png')
        depth = depth.astype(np.float32) / 65535 * 15
        mask = self.imgs[int(img_id)][..., -1]
        return depth, mask

    def get_mask(self, img_id):
        raise NotImplementedError


def parse_database_name(database_name: str) -> BaseDatabase:
    name2database = {
        'syn': GlossySyntheticDatabase,
        'real': GlossyRealDatabase,
        'nerf': NeRFSyntheticDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError


def get_database_split(database: BaseDatabase, split_type='validation'):
    if split_type == 'validation':
        random.seed(6033)
        img_ids = database.get_img_ids()
        random.shuffle(img_ids)
        test_ids = img_ids[:1]
        train_ids = img_ids[1:]
    elif split_type == 'test':
        database_name = database.database_name
        assert database_name.startswith('render')
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids


def get_database_eval_points(database):
    if isinstance(database, GlossySyntheticDatabase):
        fn = f'{database.root}/eval_pts.ply'
        if os.path.exists(fn):
            pcd = o3d.io.read_point_cloud(str(fn))
            return np.asarray(pcd.points)
        _, test_ids = get_database_split(database, 'test')
        pts = []
        for img_id in test_ids:
            depth, mask = database.get_depth(img_id)
            K = database.get_K(img_id)
            pts_ = mask_depth_to_pts(mask, depth, K)
            pose = pose_inverse(database.get_pose(img_id))
            pts_ = pose_apply(pose, pts_)
            pts.append(pts_)
        pts = np.concatenate(pts, 0).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(fn, downpcd)
        print(f'point number {len(downpcd.points)} ...')
        return np.asarray(downpcd.points, np.float32)
    else:
        raise NotImplementedError
