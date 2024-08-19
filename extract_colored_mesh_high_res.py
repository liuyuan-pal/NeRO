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

    # Load the existing mesh
    mesh = trimesh.load(cfg['mesh'], process=False)

    # Get vertices
    vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()

    # Predict materials for all vertices
    print("Predicting materials...")
    batch_size = 8192
    materials = []
    for i in range(0, vertices.shape[0], batch_size):
        batch = vertices[i:i+batch_size]
        with torch.no_grad():
            batch_materials = network.shader_network.predict_materials_n2m(batch)
        materials.append(batch_materials.cpu().numpy())
    materials = np.concatenate(materials, axis=0)

    # Split materials into albedo, metallic, and roughness
    albedo = materials[:, :3]
    metallic = materials[:, 3]
    roughness = materials[:, 4]

    # Create a new mesh with materials
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=(albedo * 255).astype(np.uint8)
    )

    # Save the colored mesh
    output_dir = Path('data/meshes')
    output_dir.mkdir(exist_ok=True, parents=True)
    colored_mesh.export(str(output_dir/f'{cfg["name"]}-{step}_colored_high_res.ply'))

    # Save material properties separately
    np.save(str(output_dir/f'{cfg["name"]}-{step}_metallic_high_res.npy'), metallic)
    np.save(str(output_dir/f'{cfg["name"]}-{step}_roughness_high_res.npy'), roughness)
    np.save(str(output_dir/f'{cfg["name"]}-{step}_albedo_high_res.npy'), albedo)

    print(f"High-resolution colored mesh and material properties saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    flags = parser.parse_args()
    main()