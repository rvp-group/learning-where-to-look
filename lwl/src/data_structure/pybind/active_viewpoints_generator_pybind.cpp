// pybind11
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// std stuff
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "../active_camera_wrapper.cuh"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(pyactivegrid, m) {
  auto active_grid = py::class_<ds::ActiveCameraGridWrapper>(m, "ActiveGrid");
  active_grid.def(py::init<>())
    .def("setCamera",
         &ds::ActiveCameraGridWrapper::setCamera,
         "set image dim, min and max depth, and pinhole camera intrinsics (focal lengths and principal points)",
         py::arg("rows"),
         py::arg("cols"),
         py::arg("min_depth"),
         py::arg("max_depth"),
         py::arg("fx"),
         py::arg("fy"),
         py::arg("cx"),
         py::arg("cy"))
    .def("setBucketExtents",
         &ds::ActiveCameraGridWrapper::setBucketExtents,
         "set bucket extents, how big buckets are, dimension may be different along axis",
         py::arg("bucket_extents"))
    .def("setNumSamples",
         &ds::ActiveCameraGridWrapper::setNumSamples,
         "set number of samples for spherical subsampling for each view",
         py::arg("num_samples"))
    .def("setGridDimension", &ds::ActiveCameraGridWrapper::setGridDimension, "set 3D grid dimension", py::arg("grid_dimension"))
    .def("setGridMinAndMax",
         &ds::ActiveCameraGridWrapper::setGridMinAndMax,
         "set dimension of 3D grid, minimum and maximum bucket",
         py::arg("min_grid_pos"),
         py::arg("max_grid_pos"))
    .def("setSparseLandmarks",
         &ds::ActiveCameraGridWrapper::setSparseLandmarks,
         "set sparse landmarks, each element numpy array 5x1",
         py::arg("landmarks"))
    .def("compute", &ds::ActiveCameraGridWrapper::compute)
    .def("getPose", &ds::ActiveCameraGridWrapper::getPose, "get grid pose, get camera in world pose from idx", py::arg("pose"))
    .def("getBestViewingDirections",
         &ds::ActiveCameraGridWrapper::getBestViewingDirections,
         "get best viewing directions from world camera pose",
         py::arg("index"),
         py::arg("max_num_directions"))
    .def("getBestViewingHits",
         &ds::ActiveCameraGridWrapper::getBestViewingHits,
         "get best viewing directions reprojected points, DEBUG method",
         py::arg("index"),
         py::arg("max_num_directions"))
    .def("getPoints2D3D",
         &ds::ActiveCameraGridWrapper::getPoints2D3D,
         "get projected features, landmark 3d and reconstruction error",
         py::arg("dir"),
         py::arg("pose"))
    .def("size", &ds::ActiveCameraGridWrapper::size, "container size");
}
