import math
import bpy
import numpy as np
import os

def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],[m01 + m10, m11 - m00 - m22, 0.0, 0.0],[m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],[m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def set_camera_by_pose(camera, pose):
    # seems the coordinate of blender is aligned by y+ to z and x+ to x+
    R_blender = np.asarray([[1,0,0],
                            [0,0,-1],
                            [0,1,0]]) # x_blend = R_blender @ x_wrd
    cam_pt = (pose[:,:3].T @ -pose[:,3:])[...,0] # 3 in x_wrd
    cam_rot = pose[:,:3]
    # x_cam = R_rot @ x_wrd = R @ R_blender.T @ x_blender
    cam_rot = cam_rot @ R_blender.T
    cam_pt = R_blender @ cam_pt
    cam_rot = np.diag([1,-1,-1]) @ cam_rot

    camera.location[0] = cam_pt[0]
    camera.location[1] = cam_pt[1]
    camera.location[2] = cam_pt[2]
    camera.rotation_mode = 'QUATERNION'
    q = quaternion_from_matrix(cam_rot.T)
    camera.rotation_quaternion[0] = q[0]
    camera.rotation_quaternion[1] = q[1]
    camera.rotation_quaternion[2] = q[2]
    camera.rotation_quaternion[3] = q[3]

def add_env_light(fn):
    world_tree = bpy.context.scene.world.node_tree
    env_node = world_tree.nodes.new(type='ShaderNodeTexEnvironment')
    back_node = world_tree.nodes['World Output']
    world_tree.links.new(env_node.outputs['Color'], back_node.inputs['Surface'])
    bpy.ops.image.open(filepath=os.path.abspath(fn))
    env_node.image = bpy.data.images[fn.split(os.path.sep)[-1]]

def pose_inverse(pose):
    R = pose[:,:3].T
    t = - R @ pose[:,3:]
    return np.concatenate([R,t],-1)

def az_el_to_points(azimuths, elevations):
    z = np.sin(elevations)
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    return np.stack([x,y,z],-1) #

def look_at_with_up(view_pts,center,up):
    up = up/np.linalg.norm(up) # 3
    view_dir = center[None,:]-view_pts
    view_dir /= np.linalg.norm(view_dir,2,1,keepdims=True)
    z_axis = view_dir # n,3
    y_axis = up[None,:] - np.sum(view_dir * up[None,:], 1,keepdims=True) * view_dir # n,3
    y_axis = -y_axis
    y_axis /= np.linalg.norm(y_axis,2,1,keepdims=True)
    x_axis = np.cross(y_axis,z_axis) # n,3
    rotation = np.stack([x_axis,y_axis,z_axis],2).transpose([0,2,1])
    return rotation

def generate_relghting_poses(num, azimuth, elevation, dist):
    num = num
    begin_az = azimuth
    el = elevation
    dist = dist
    az = np.deg2rad(begin_az) + np.linspace(-np.pi/2, np.pi/2, num)
    el = np.ones_like(az) * np.deg2rad(el)

    cam_pts = az_el_to_points(az, el)
    R_trans = np.asarray([[1,0,0],[0,0,-1], [0,1,0]]) # x_norm = R_trans @ x_wrd
    cam_rots = look_at_with_up(cam_pts, np.asarray([0,0,0],np.float32), np.asarray([0,0,1],np.float32)) # R_cam @ x_norm = x_cam
    cam_rots = cam_rots @ R_trans[None,:,:] # R_cam @ R_trans @ x_wrd = x_cam
    cam_trans = np.asarray([0, 0, dist])[None,:,None] # 1,3,1
    cam_trans = np.repeat(cam_trans,num,0) # 32,3,1
    poses = np.concatenate([cam_rots,cam_trans],-1)
    return poses

def add_env_light(fn):
    world_tree = bpy.context.scene.world.node_tree
    env_node = world_tree.nodes.new(type='ShaderNodeTexEnvironment')
    back_node = world_tree.nodes['World Output']
    world_tree.links.new(env_node.outputs['Color'], back_node.inputs['Surface'])
    bpy.ops.image.open(filepath=os.path.abspath(fn))
    env_node.image = bpy.data.images[fn.split(os.path.sep)[-1]]


def setup(h, w, tile_size=4096, samples=4096):
    # set environment
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.render.tile_x = 128 # use gpu to speed up rendering
    # bpy.context.scene.render.tile_y = 128
    bpy.context.scene.cycles.tile_size = tile_size
    bpy.context.scene.cycles.samples = samples

    # set output settings
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # delete default box
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
    # delete default light
    bpy.data.objects['Light'].select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
