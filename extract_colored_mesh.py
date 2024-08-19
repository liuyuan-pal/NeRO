import argparse
from pathlib import Path
import torch
import trimesh
import numpy as np
from utils.base_utils import load_cfg
from network.renderer import NeROMaterialRenderer

def main():
    cfg = load_cfg(flags.cfg)
    network = NeROMaterialRenderer(cfg, is_train=False)
    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    step = ckpt['step']
    network.load_state_dict(ckpt['network_state_dict'])
    network.eval().cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'Successfully loaded {cfg["name"]} step {step}!')

    # Predict materials
    with torch.no_grad():
        materials = network.predict_materials()

    # Get the mesh from the network
    mesh = network.mesh

    # Convert Open3D mesh to trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Create a new trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Add vertex colors albedo
    albedo_colors = (materials['albedo'] * 255).astype(np.uint8)
    tri_mesh.visual.vertex_colors = albedo_colors

    # Save the colored mesh
    output_dir = Path('data/meshes')
    output_dir.mkdir(exist_ok=True)
    tri_mesh.export(str(output_dir/f'{cfg["name"]}-{step}_colored.ply'))

    # Save material properties separately
    np.save(str(output_dir/f'{cfg["name"]}-{step}_metallic.npy'), materials['metallic'])
    np.save(str(output_dir/f'{cfg["name"]}-{step}_roughness.npy'), materials['roughness'])
    np.save(str(output_dir/f'{cfg["name"]}-{step}_albedo.npy'), materials['albedo'])

    print(f"Colored mesh and material properties saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    flags = parser.parse_args()
    main()