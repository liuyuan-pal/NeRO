#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace raytracing {

// abstract class of raytracer
class RayTracer {
public:
    RayTracer() {}
    virtual ~RayTracer() {}

    virtual void trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor normals, at::Tensor depth) = 0;
};

// function to create an implementation of raytracer
RayTracer* create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles);
    
} // namespace raytracing