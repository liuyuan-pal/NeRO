import argparse
from pathlib import Path

import numpy as np
import torch

from network.renderer import NeROMaterialRenderer
from utils.base_utils import load_cfg
from utils.raw_utils import linear_to_srgb


def main():
    cfg = load_cfg(flags.cfg)
    network = NeROMaterialRenderer(cfg, False)

    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'successfully load {cfg["name"]} step {step}!')

    with torch.no_grad():
        material_dir = f'data/materials/{cfg["name"]}-{step}'
        Path(material_dir).mkdir(parents=True, exist_ok=True)
        materials = network.predict_materials()

        print(
            'warning!!!!! we transform both albedo/metallic/roughness with gamma correction because our blender script uses vertex colors to store them, '
            'it seems blender will apply an inverse gamma correction so that the results will be incorrect without this gamma correct\n'
            'for more information refer to https://blender.stackexchange.com/questions/87576/vertex-colors-loose-color-data/87583#87583')
        np.save(f'{material_dir}/metallic.npy', linear_to_srgb(materials['metallic']))
        np.save(f'{material_dir}/roughness.npy', linear_to_srgb(materials['roughness']))
        np.save(f'{material_dir}/albedo.npy', linear_to_srgb(materials['albedo']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    flags = parser.parse_args()
    main()
