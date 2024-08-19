#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <cstdint>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <Eigen/Dense>

namespace raytracing {

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT2 = 1.41421356237309504880f;


// enum class EMeshSdfMode : int {
// 	Watertight,
// 	Raystab,
// 	PathEscape,
// };
// static constexpr const char* MeshSdfModeStr = "Watertight\0Raystab\0PathEscape\0\0";

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
    if (n_elements <= 0) {
        return;
    }
    kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}

inline __host__ __device__ float sign(float x) {
    return copysignf(1.0, x);
}

template <typename T>
__host__ __device__ T clamp(T val, T lower, T upper) {
    return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
__host__ __device__ void host_device_swap(T& a, T& b) {
    T c(a); a=b; b=c;
}

}