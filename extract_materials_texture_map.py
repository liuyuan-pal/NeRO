import argparse
from pathlib import Path

import numpy as np
import torch

from network.renderer import NeROMaterialRenderer, NeROShapeRenderer
from utils.base_utils import load_cfg
from utils.raw_utils import linear_to_srgb

##Edit for material weights to texture map extraction
import xatlas
import nvdiffrast.torch as dr
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
import cv2
import os
import torch.nn as nn

def contract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    return xyzs

def texture_mapper(path = 'n2m_test', h0=1024, w0=1024):
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
    
    ##Original Network Settings
    cfg = load_cfg(flags.cfg)
    network = NeROMaterialRenderer(cfg, False)

    ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval().cuda()

    ##nerf2mesh extract settings
    device = 'cuda'
    glctx = dr.RasterizeGLContext(output_db=False)

    vertices = []
    triangles = []
    v_cumsum = [0]
    f_cumsum = [0]

    vertices.append(network.tri_mesh.vertices)
    triangles.append(network.tri_mesh.faces + v_cumsum[-1])

    vertices = np.concatenate(vertices, axis=0)
    triangles = np.concatenate(triangles, axis=0)
    v_cumsum = np.array(v_cumsum)
    f_cumsum = np.array(f_cumsum)

    vertices = torch.from_numpy(vertices).float().cuda()  # [N, 3]
    triangles = torch.from_numpy(triangles).int().cuda()
    vertices_offsets = nn.Parameter(torch.zeros_like(vertices))
    def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
        # v, f: torch Tensor

        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

        # unwrap uv in contracted space
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)

        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]


        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
        print(f'[INFO] numpy to torch')
        # render uv maps
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        print(f'[INFO] diffrast start')
        rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

        print(f'[INFO] diffrast done')
        # masked query
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        feats = np.zeros([h * w, 5])
        normals = np.zeros([h * w, 3])

        print(f'[INFO] inference start')
        if mask.any():
            xyzs = xyzs[mask]  # [M, 3]

            # batched inference to avoid OOM
            all_feats = []
            all_norms = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                with torch.cuda.amp.autocast(enabled=True):
                    points = xyzs[head:tail]
                    all_feats.append(network.shader_network.predict_materials_n2m(points).float().detach().cpu().numpy())
                head += 640000

            mask_cpu = mask.cpu().numpy()
            feats[mask_cpu] = np.concatenate(all_feats)

        print(f'[INFO] inference done')
        feats = feats.reshape(h, w, -1)  # 6 channels
        mask_cpu = mask_cpu.reshape(h, w)
        feats = linear_to_srgb(feats)
        feats = (feats * 255).astype(np.uint8)


        inpaint_region = binary_dilation(mask_cpu, iterations=32)  # pad width
        inpaint_region[mask_cpu] = 0

        search_region = mask_cpu.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

        # do ssaa after the NN search, in numpy
        feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo
        feats1 = cv2.cvtColor(feats[..., 3], cv2.COLOR_GRAY2BGR)  # metallic
        feats2 = cv2.cvtColor(feats[..., 4], cv2.COLOR_GRAY2BGR)  # roughness


        if ssaa > 1:
            feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
            feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)
            feats2 = cv2.resize(feats2, (w0, h0), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
        cv2.imwrite(os.path.join(path, f'feat1_{cas}.jpg'), feats1)
        cv2.imwrite(os.path.join(path, f'feat2_{cas}.jpg'), feats2)

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'mesh_{cas}.obj')
        mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

        print(f'[INFO] writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:

            fp.write(f'mtllib mesh_{cas}.mtl \n')

            print(f'[INFO] writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            print(f'[INFO] writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            print(f'[INFO] writing faces {f_np.shape}')
            fp.write(f'usemtl defaultMat \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1 1 1 \n')
            fp.write(f'Kd 1 1 1 \n')
            fp.write(f'Ks 0 0 0 \n')
            fp.write(f'Tr 1 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0 \n')
            fp.write(f'map_Kd feat0_{cas}.jpg \n')

        return xyzs, mask

    v = (vertices + vertices_offsets).detach()
    f = triangles.detach()

    return _export_obj(v, f, h0, w0, 2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    flags = parser.parse_args()
    rast_xys, rast_mask = texture_mapper()
