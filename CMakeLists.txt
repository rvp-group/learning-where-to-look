cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(lwl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# eigen 
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# python bindings
find_package(pybind11 REQUIRED)

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# cuda
find_file(CUDA_HEADER cuda_runtime.h PATHS /usr/include/ /usr/local/cuda/include/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${CUDA_HEADER}")

find_package(CUDA REQUIRED)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
# just warning suppression does not work!
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo -Wno-dev") # -Wno-dev no suppress CUDA_ARCHITECTURES warning
# disabling stupid eigen warnings, not working https://github.com/cms-sw/cmssw/issues/33369
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --nv-diag-suppress 20014")  

message("CUDA ARCH FLAGS: ${ARCH_FLAGS}")
message("CUDA FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

# resolve collision definition, hard coded for now beteer to make it!
# add_definitions(-DRESOLVE_COLLISION)

# gtests
find_package(GTest REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/lwl/src
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIR}
)

add_subdirectory(lwl)