#pragma once
#include "cuda_utils.cuh"
#include <cstring>
#include <iterator>

namespace ds {

  enum MemType { Host = 0, Device = 1 };

  /**
   * DualMatrix_ class to handle easily matrix type bouncing from GPU to RAM
   */
  template <typename CellType_>
  class __align__(16) DualMatrix_ {
  public:
    using ThisType = DualMatrix_<CellType_>;
    using CellType = CellType_;

    __host__ explicit DualMatrix_(const uint16_t rows, const uint16_t cols) {
      CUDA_CHECK(cudaMalloc((void**) &device_instance_, sizeof(ThisType)));
      resize(rows, cols);
    }

    __host__ explicit DualMatrix_() : buffers_{nullptr, nullptr}, device_instance_(nullptr), rows_(0), cols_(0), capacity_(0) {
      CUDA_CHECK(cudaMalloc((void**) &device_instance_, sizeof(ThisType)));
      CUDA_CHECK(cudaMemcpy(device_instance_, this, sizeof(ThisType), cudaMemcpyHostToDevice));
    }

    __host__ DualMatrix_(const DualMatrix_& src_) : DualMatrix_(src_.rows_, src_.cols_) {
      memcpy(buffers_[Host], src_.buffers_[Host], sizeof(CellType) * capacity_);
      CUDA_CHECK(cudaMemcpy(buffers_[Device], src_.buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToDevice));
    }

    DualMatrix_& operator=(const DualMatrix_& src_) {
      resize(src_.rows_, src_.cols_);
      memcpy(buffers_[Host], src_.buffers_[Host], sizeof(CellType) * capacity_);
      CUDA_CHECK(cudaMemcpy(buffers_[Device], src_.buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToDevice));
      return *this;
    }

    ~DualMatrix_() {
      if (device_instance_)
        cudaFree(device_instance_);
      clearHostBuffer();
      clearDeviceBuffer();
    }

    __host__ inline void resize(const uint16_t rows, const uint16_t cols) {
      // if size is ok, do nothing
      if (rows == rows_ && cols == cols_)
        return;
      rows_ = rows;
      cols_ = cols;
      sync_();
    }

    // clang-format off
    __host__ __device__ inline ThisType* deviceInstance() { return device_instance_; }
    __host__ inline const ThisType* deviceInstance() const { return device_instance_; } 
    __host__ void fill(const CellType& value_, const bool device_only_ = false);
    __host__ inline const uint32_t nThreads() const { return n_threads_; }
    __host__ inline const uint32_t nBlocks() const { return n_blocks_; }
    __host__ __device__ inline const uint16_t rows() const { return rows_; }
    __host__ __device__ inline const uint16_t cols() const { return cols_; }
    __host__ __device__ inline const uint32_t size() const { return capacity_; }
    __host__ __device__ inline const bool empty() const { return capacity_ == 0; };
    // clang-format on

    __host__ __device__ inline bool inside(const uint16_t row, const uint16_t col) const {
      return row >= 0 && col >= 0 && row < rows_ && col < cols_;
    }

    __host__ __device__ inline bool onBorder(const uint16_t row, const uint16_t col) const {
      return row == 0 || col == 0 || row == rows_ - 1 || col == cols_ - 1;
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& at(const uint16_t index_) const {
      return buffers_[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& at(const uint16_t index_) {
      return buffers_[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& at(const uint16_t row, const uint16_t col) const {
      return buffers_[MemType][row * cols_ + col];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& at(const uint16_t row, const uint16_t col) {
      return buffers_[MemType][row * cols_ + col];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& operator()(const uint16_t row, const uint16_t col) {
      return buffers_[MemType][row * cols_ + col];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType& operator[](const uint16_t index_) {
      return buffers_[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType& operator[](const uint16_t index_) const {
      return buffers_[MemType][index_];
    }

    template <int MemType = 0>
    __host__ __device__ inline const CellType* data() const {
      return buffers_[MemType];
    }

    template <int MemType = 0>
    __host__ __device__ inline CellType* data() {
      return buffers_[MemType];
    }

    // copy whole device buffer to host, for debugging at the moment
    __host__ inline void toHost() {
      CUDA_CHECK(cudaMemcpy(buffers_[Host], buffers_[Device], sizeof(CellType) * capacity_, cudaMemcpyDeviceToHost));
    }

    __host__ inline void toDevice() {
      CUDA_CHECK(cudaMemcpy(buffers_[Device], buffers_[Host], sizeof(CellType) * capacity_, cudaMemcpyHostToDevice));
    }

    // ! initialize both buffers, similar to fill but not relying on CUDA kernels
    __host__ inline void initialize(const CellType& value) {
      for (int i = 0; i < capacity_; ++i) {
        buffers_[Host][i] = value;
      }
      toDevice();
    }

    __host__ inline void clearDeviceBuffer() {
      if (buffers_[Device]) {
        cudaFree(buffers_[Device]);
        buffers_[Device] = nullptr;
      }
    }

    __host__ inline void clearHostBuffer() {
      if (buffers_[Host]) {
        delete[] buffers_[Host];
        buffers_[Host] = nullptr;
      }
    }

  protected:
    inline void sync_() {
      if (capacity_ == (uint32_t)(rows_ * cols_)) {
        copyHeader_();
        return;
      }

      if (buffers_[Device]) {
        cudaFree(buffers_[Device]);
        buffers_[Device] = nullptr;
      }

      if (buffers_[Host]) {
        delete[] buffers_[Host];
        buffers_[Host] = nullptr;
      }
      capacity_ = (uint32_t)(rows_ * cols_);
      if (capacity_) {
        buffers_[Host] = new CellType[capacity_];
        CUDA_CHECK(cudaMalloc((void**) &buffers_[Device], sizeof(CellType) * capacity_));
        CUDA_CHECK(cudaMemcpy(buffers_[Device], buffers_[Host], sizeof(CellType) * capacity_, cudaMemcpyHostToDevice));
      }

      copyHeader_();
    }

    inline void copyHeader_() {
      n_threads_ = N_THREADS;
      n_blocks_  = (capacity_ + n_threads_ - 1) / n_threads_;
      // once class fields are populated copy ptr on device
      CUDA_CHECK(cudaMemcpy(device_instance_, this, sizeof(ThisType), cudaMemcpyHostToDevice));
    }

    CellType* buffers_[2]      = {nullptr, nullptr};
    ThisType* device_instance_ = nullptr;
    uint16_t rows_             = 0;
    uint16_t cols_             = 0;
    uint32_t capacity_         = 0;
    uint32_t n_threads_        = 0;
    uint32_t n_blocks_         = 0;
  };

  using CUDAMatrixf   = DualMatrix_<float>;
  using CUDAMatrixf3  = DualMatrix_<float3>;
  using CUDAMatrixb   = DualMatrix_<bool>;
  using CUDAMatrixuc3 = DualMatrix_<uchar3>;

} // namespace ds
