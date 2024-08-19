#pragma once

#include <raytracing/raytracer.h>

#include <raytracing/common.h>
#include <raytracing/bvh.cuh>

#include <Eigen/Dense>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace raytracing {

class RayTracerImpl : public RayTracer {
public:

    // accept numpy array (cpu) to init 
    RayTracerImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : RayTracer() {

        const size_t n_vertices = vertices.rows();
        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2))};
        }

        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }

        triangle_bvh->build(triangles_cpu, 8);

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);

        // TODO: need OPTIX
        // triangle_bvh->build_optix(triangles_gpu, m_inference_stream);

    }

    // accept torch tensor (gpu) to init
    void trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor normals, at::Tensor depth) {

        // must be contiguous, float, cuda, shape [N, 3]. check in torch side.

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), normals.data_ptr<float>(), depth.data_ptr<float>(), triangles_gpu.data(), stream);
    }

    std::vector<Triangle> triangles_cpu;
    GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
RayTracer* create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new RayTracerImpl{vertices, triangles};
}

} // namespace raytracing