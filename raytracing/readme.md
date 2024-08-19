# RayTracer

A CUDA Mesh RayTracer with BVH acceleration.

### Install

```python
git clone https://github.com/ashawkey/raytracing
cd raytracing
pip install .
```

### Usage

Example for a mesh normal renderer:

```bash
python renderer.py # default, show a dodecahedron
python renderer.py --mesh example.ply # show any mesh file
```

https://user-images.githubusercontent.com/25863658/183238748-7ac82808-6cd3-4bb6-867a-9c22f8e3f7dd.mp4

Example code:

```python
import numpy as np
import trimesh

import torch
import raytracing

# build BVH from mesh
mesh = trimesh.load('example.ply')
RT = raytracing.RayTracer(mesh.vertices, mesh.faces) # build with numpy.ndarray

# get rays
rays_o, rays_d = get_ray(pose, intrinsics, H, W) # [N, 3], [N, 3], query with torch.Tensor (on cuda)

# query ray-mesh intersection
intersections, face_normals, depth = RT.trace(rays_o, rays_d) # [N, 3], [N, 3], [N,]
```



### Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp)!
