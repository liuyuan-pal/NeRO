#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <torch/extension.h>

#include <raytracing/raytracer.h>


namespace py = pybind11;
using namespace raytracing;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

py::class_<RayTracer>(m, "RayTracer")
    .def("trace", &RayTracer::trace);

m.def("create_raytracer", &create_raytracer);

}