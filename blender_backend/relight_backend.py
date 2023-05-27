import argparse
import os
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

print(os.path.abspath('.'))
sys.path.append(os.path.abspath('.'))
from blender_backend.blender_utils import setup, set_camera_by_pose, generate_relghting_poses, add_env_light

import bpy


def render():
    args.output = os.path.abspath(args.output)
    args.env_fn = os.path.abspath(args.env_fn)

    Path(args.output).mkdir(exist_ok=True, parents=True)
    setup(args.height, args.width, tile_size=256**2, samples=args.samples)
    bpy.context.scene.render.film_transparent = True

    bpy.ops.import_mesh.ply(filepath=args.mesh)
    obj = bpy.data.objects[Path(args.mesh).stem]

    metallic = np.load(f'{args.material}/metallic.npy')
    roughness = np.load(f'{args.material}/roughness.npy')
    albedo = np.load(f'{args.material}/albedo.npy')

    mat_vert_color = obj.data.vertex_colors.new()
    rgb_vert_color = obj.data.vertex_colors.new()

    # a map from the unique index to the loop index
    vertex_map = defaultdict(list)
    for poly in obj.data.polygons:
        for v_ix, l_ix in zip(poly.vertices, poly.loop_indices):
            vertex_map[v_ix].append(l_ix)

    # set all loop index
    for v_ix, l_ixs in vertex_map.items():
        for l_ix in l_ixs:
            rgb_vert_color.data[l_ix].color.data.color[:3] = albedo[v_ix]
            mat_vert_color.data[l_ix].color.data.color[0] = metallic[v_ix,0]
            mat_vert_color.data[l_ix].color.data.color[1] = roughness[v_ix,0]

    if args.trans:
        # trans = np.asarray([[1,0,0],[0,0,-1],[0,1,0]],np.float32)
        obj.rotation_euler[0]=np.pi/2

    # create a material
    material = bpy.data.materials.new(name='mat')
    material.use_nodes = True
    obj.data.materials.append(material)
    bsdf_node = material.node_tree.nodes['Principled BSDF']
    bsdf_node.inputs['Specular'].default_value = 0.5
    bsdf_node.inputs['Specular Tint'].default_value = 0.0
    bsdf_node.inputs['Sheen Tint'].default_value = 0.0
    bsdf_node.inputs['Clearcoat Roughness'].default_value = 0.0

    # link base color
    color_node = material.node_tree.nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = rgb_vert_color.name
    material.node_tree.links.new(color_node.outputs['Color'], bsdf_node.inputs['Base Color'])

    # link metallic and roughness
    mr_node = material.node_tree.nodes.new("ShaderNodeVertexColor")
    mr_node.layer_name = mat_vert_color.name
    sep_node = material.node_tree.nodes.new("ShaderNodeSeparateRGB")
    material.node_tree.links.new(mr_node.outputs['Color'], sep_node.inputs['Image'])
    material.node_tree.links.new(sep_node.outputs['R'], bsdf_node.inputs['Metallic'])
    material.node_tree.links.new(sep_node.outputs['G'], bsdf_node.inputs['Roughness'])

    # add background light
    print('load env map ...')
    add_env_light(fn=args.env_fn)

    camera = bpy.data.objects['Camera']
    cam_poses = generate_relghting_poses(args.num, args.azimuth, args.elevation, args.cam_dist)

    print('rendering ...')
    for k in range(args.num):
        if os.path.exists(f'{args.output}/{k}.png'): continue
        bpy.context.scene.render.filepath = f'{args.output}/{k}'
        set_camera_by_pose(camera, cam_poses[k])
        bpy.ops.render.render(write_still=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--output', type=str, default='data/relight')
    parser.add_argument('--env_fn', type=str, default='data/hdr/')
    parser.add_argument('--mesh', type=str, default='data/meshes/bell_shape-300000.ply')
    parser.add_argument('--material', type=str, default='data/materials/bell_material-100000')

    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--cam_dist', type=float, default=3.0)
    parser.add_argument('--num', type=int, default=360)

    parser.add_argument('--trans', action='store_true', dest='trans', default=False)

    parser.add_argument('--pose_type', type=str, default='video')
    parser.add_argument('--azimuth', type=float, default=0.0)
    parser.add_argument('--elevation', type=float, default=45.0)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    render()
