
#include "cuda_matrix.cuh"
#include <opencv2/opencv.hpp>

namespace ds {
  // ! template class instantiation with nvcc can only be solved by including this cu file where needed

  template <typename CellType_>
  __global__ void fill_kernel(CellType_* data_device, const CellType_ value, const uint32_t capacity) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < capacity)
      data_device[tid] = value;
  }

  template <typename CellType_>
  void DualMatrix_<CellType_>::fill(const CellType_& value, const bool device_only) {
    fill_kernel<<<n_blocks_, n_threads_>>>(buffers_[Device], value, capacity_);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (device_only)
      return;
    // fill host and then fill device
    for (int i = 0; i < capacity_; ++i) {
      buffers_[Host][i] = value;
    }
  }

} // namespace ds
