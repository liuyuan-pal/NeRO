from pathlib import Path

import numpy as np
from skimage.io import imsave

from network.loss import Loss
from utils.base_utils import color_map_backward
from utils.draw_utils import concat_images_list
from skimage.metrics import structural_similarity


def compute_psnr(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


def process_key_img(data, h, w):
    img = color_map_backward(data.detach().cpu().numpy())
    img = img.reshape([h, w, -1])
    if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
    return img


def get_key_images(data_pr, keys, h, w):
    results = []
    for k in keys:
        if k in data_pr: results.append(process_key_img(data_pr[k], h, w))
    return results


def draw_materials(data_pr, h, w):
    keys = ['diffuse_albedo', 'diffuse_light', 'diffuse_color',
            'specular_albedo', 'specular_light', 'specular_color', 'specular_ref',
            'metallic', 'roughness', 'occ_prob', 'indirect_light']
    results = get_key_images(data_pr, keys, h, w)
    results = [concat_images_list(*results[0:3]), concat_images_list(*results[3:7]), concat_images_list(*results[7:])]
    return results


class ShapeRenderMetrics(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = color_map_backward(data_pr['gt_rgb'].detach().cpu().numpy())  # h,w,3
        imgs = [rgb_gt]

        # compute psnr
        rgb_pr = color_map_backward(data_pr['ray_rgb'].detach().cpu().numpy())  # h,w,3
        psnr = compute_psnr(rgb_gt, rgb_pr)
        ssim = structural_similarity(rgb_gt, rgb_pr, win_size=11, channel_axis=2, data_range=255)
        outputs = {'psnr': np.asarray([psnr]), 'ssim': np.asarray([ssim])}
        imgs.append(rgb_pr)

        # normal
        h, w, _ = rgb_pr.shape
        normal = color_map_backward(data_pr['normal'].detach().cpu().numpy())  # h,w,3
        imgs.append(normal.reshape([h, w, 3]))

        if 'human_light' in data_pr:
            imgs.append(process_key_img(data_pr['human_light'], h, w))

        imgs = [concat_images_list(*imgs)]

        imgs += draw_materials(data_pr, h, w)

        # output image
        data_index = kwargs['data_index']
        model_name = kwargs['model_name']
        output_path = Path(f'data/train_vis/{model_name}')
        output_path.mkdir(exist_ok=True, parents=True)
        imsave(f'{str(output_path)}/{step}-index-{data_index}.jpg', concat_images_list(*imgs, vert=True))
        return outputs


class MaterialRenderMetrics(Loss):
    def __init__(self, cfg):
        pass

    def __call__(self, data_pr, data_gt, step, *args, **kwargs):
        rgb_gt = color_map_backward(data_pr['rgb_gt'].detach().cpu().numpy())  # h,w,3
        rgb_pr = color_map_backward(data_pr['rgb_pr'].detach().cpu().numpy())  # h,w,3
        imgs = [rgb_gt, rgb_pr]

        # compute psnr
        psnr = compute_psnr(rgb_gt, rgb_pr)
        ssim = structural_similarity(rgb_gt, rgb_pr, win_size=11, channel_axis=2, data_range=255)
        outputs = {'psnr': np.asarray([psnr]), 'ssim': np.asarray([ssim])}

        additional_keys = ['albedo', 'metallic', 'roughness', 'specular_light', 'specular_color', 'diffuse_light',
                           'diffuse_color']
        for k in additional_keys:
            img = color_map_backward(data_pr[k].detach().cpu().numpy())
            if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
            imgs.append(img)

        output_imgs = [concat_images_list(*imgs[:5]), concat_images_list(*imgs[5:])]

        # output image
        data_index = kwargs['data_index']
        model_name = kwargs['model_name']
        output_path = Path(f'data/train_vis/{model_name}')
        output_path.mkdir(exist_ok=True, parents=True)
        imsave(f'{str(output_path)}/{step}-index-{data_index}.jpg', concat_images_list(*output_imgs, vert=True))
        return outputs


name2metrics = {
    'shape_render': ShapeRenderMetrics,
    'mat_render': MaterialRenderMetrics,
}


def psnr(results):
    return np.mean(results['psnr'])


name2key_metrics = {
    'psnr': psnr,
}
