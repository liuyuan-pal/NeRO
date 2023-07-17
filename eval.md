
### Evaluate the Glossy Synthetic dataset
Ensure you have the following files
```
NeRO
|-- data
    |-- GlossySynthetic
        |-- bell
            ...
    |-- meshes
        | -- bell_shape-300000.ply
             ...
```
Evaluate the mesh quality in chamfer distance by
```shell
python eval_synthetic_shape.py --object bell --mesh data/meshes/bell_shape-300000.ply
```
Then, you should see the printed chamfer distances.

### Evaluate the Glossy Real dataset
1. We need to meshes that do not belong to the object. We use the MeshLab for this task.
2. Then, we save the processed 3D mesh and then use CloudCompare to sample points on the surface.
3. Finally, we save the cropped point cloud in the `data/point_cloud/bear-nero.ply`. We sample 500k points on the mesh in default.
4. Similarly, we sample a point cloud on the gt mesh and save it to `data/point_cloud/bear-gt.ply`.
5. You may download all sampled point clouds at [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/Ee2M6tZZ22tOnEjCo_yxVywB2jqgbjTNADJPIr_POuaa0g?e=EcnXCk).
6. Eval the chamfer distance by
```shell
python eval_real_shape.py --pr data/point_cloud/bear-pr.ply --gt data/point_cloud/bear-gt.ply
```
Then, you should see the computed chamfer distance.