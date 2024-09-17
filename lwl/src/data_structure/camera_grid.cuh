#pragma once
#include "camera.cuh"
#include "camera_active.cuh"
#include "cuda_algebra.cuh"
#include "cuda_math.cuh"
#include "cuda_matrix.cuh"
#include "cuda_matrix_conversion.cuh"
#include <Eigen/Dense>
#include <vector>

namespace ds {

  struct __align__(16) Sparse {
    Sparse() {
      pos   = make_float3(0.f, 0.f, 0.f);
      error = 0.f;
      idx   = 0;
    }

    Sparse(const Eigen::Vector3f& eigpos, const float err = 0.f, const unsigned long long int i = 0) {
      pos   = Eig2CUDA(eigpos);
      error = err;
      idx   = i;
    }

    float3 pos;  // position of point in 3d
    float error; // error from reconstruction
    unsigned long long int idx;
  };

  inline Eigen::Matrix3f rot_from_azel(const float az, const float el) {
    const float c_az = cos(az);
    const float s_az = sin(az);
    const float c_el = cos(el);
    const float s_el = sin(el);
    Eigen::Matrix3f rot;
    rot << c_az, -s_az * c_el, s_az * s_el, s_az, c_az * c_el, -c_az * s_el, 0, s_el, c_el;
    return rot;
  }

  // ! main data structure for sparse 3D grid representation
  // ! each cube in space represent a camera location
  // ! perform spherical sampling to calculate num_samples bearing (optical-axis) camera directions
  template <typename ActiveCamType>
  struct __align__(16) CameraGrid_ {
    CameraGrid_(const Eigen::Vector3f& bucket_extents,  // camera blocks extents
                const Eigen::Vector3f& grid_dimensions, // total grid dim along 3-axis
                const Eigen::Vector3f& min_grid_pos,    // minimum grid position
                const Eigen::Vector3f& max_grid_pos,
                const uint num_samples = 0,
                const Camera* camera   = nullptr) :
      num_samples_(num_samples) {
      sampling_directions_ = nullptr;
      CUDA_CHECK(cudaMalloc((void**) &d_instance_, sizeof(CameraGrid_)));
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(CameraGrid_), cudaMemcpyHostToDevice));

      // eigen host stuff
      bucket_extents_  = bucket_extents;
      grid_dimensions_ = grid_dimensions;
      min_grid_pos_    = min_grid_pos;
      max_grid_pos_    = max_grid_pos;

      d_bucket_extents_  = Eig2CUDA(bucket_extents);
      d_grid_dimensions_ = Eig2CUDA(grid_dimensions);
      d_min_grid_pos_    = Eig2CUDA(min_grid_pos);
      d_max_grid_pos_    = Eig2CUDA(max_grid_pos);

      // calculating how many buckets along the grid for each dim
      dim_x_    = ((uint) d_grid_dimensions_.x / d_bucket_extents_.x + 0.5) + 1;
      dim_y_    = ((uint) d_grid_dimensions_.y / d_bucket_extents_.y + 0.5) + 1;
      dim_yx_   = dim_y_ * dim_x_; // we need to always make this product when accessing, better here
      dim_z_    = ((uint) d_grid_dimensions_.z / d_bucket_extents_.z + 0.5) + 1;
      capacity_ = dim_yx_ * dim_z_;

      // std::cerr << dim_x_ << " " << dim_y_ << " " << dim_z_ << std::endl;
      // exit(0);

      // spherical subsampling
      if (camera) {
        sampling_directions_ = new float4[num_samples_];
        for (int i = 0; i < num_samples_; ++i) {
          const float index = (float) i + 0.5f;
          const float az    = acos(1 - 2 * index / (float) num_samples_);
          const float el    = M_PI * (1 + pow(5, 0.5f)) * index;
          // const Eigen::Quaternionf q =
          //   Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitY());
          const Eigen::Matrix3f so3rot = rot_from_azel(az, el);
          const Eigen::Quaternionf q(so3rot);
          sampling_directions_[i] = make_float4(q.x(), q.y(), q.z(), q.w());
        }
      }
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(CameraGrid_), cudaMemcpyHostToDevice));
    }

    CameraGrid_() = default;

    ~CameraGrid_() {
      if (sampling_directions_)
        delete[] sampling_directions_;
      delete[] buffers_[Host];
      CUDA_CHECK(cudaFree(buffers_[Device]));
      CUDA_CHECK(cudaFree(d_instance_));
    }

    // host transformations
    inline Eigen::Vector3i const bucketToBufferPos(const Eigen::Vector3f& bucket) const {
      Eigen::Vector3f inv_bucket_extents;
      inv_bucket_extents.array() = 1.f / bucket_extents_.array();
      Eigen::Vector3f res        = ((bucket - min_grid_pos_).cwiseProduct(inv_bucket_extents)).array().round();
      return res.cast<int>();
    }

    inline Eigen::Vector3f const bufferPosToBucket(const Eigen::Vector3i& buffpos) const {
      // return min_grid_pos_.cast<float>() + buffpos.cast<float>().cwiseProduct(bucket_extents_);
      return min_grid_pos_ + buffpos.cast<float>().cwiseProduct(bucket_extents_);
    }

    // device transformations
    // ! bucket to buffer indices
    __forceinline__ __device__ int3 const bucketToBufferPos(const float3& bucket) const {
      // return roundf((bucket - make_float3(d_min_grid_pos_)) / d_bucket_extents_);
      return roundf((bucket - d_min_grid_pos_) / d_bucket_extents_);
    }

    // ! buffer indices to camera bucket
    __forceinline__ __device__ float3 const bufferPosToBucket(const int3& buffpos) {
      // return make_float3(d_min_grid_pos_) + make_float3(buffpos) * d_bucket_extents_;
      return d_min_grid_pos_ + make_float3(buffpos) * d_bucket_extents_;
    }

    __forceinline__ __device__ bool const inside(const int3& pos) {
      if (pos.x < dim_x_ && pos.y < dim_y_ && pos.z < dim_z_) // strict since we start from 0
        return true;
      return false;
    }

    __forceinline__ __device__ bool const isInsideGrid(const float3& pos) {
      if (pos.x < d_min_grid_pos_.x || pos.y < d_min_grid_pos_.y || pos.z < d_min_grid_pos_.z)
        return false;
      if (pos.x > d_max_grid_pos_.x || pos.y > d_max_grid_pos_.y || pos.z > d_max_grid_pos_.z)
        return false;
      return true;
    }

    __host__ __device__ inline uint const posToLinear(const int3& pos) const {
      return pos.z * dim_yx_ + pos.y * dim_x_ + pos.x;
    }

    template <int MemType = 0>
    __host__ __device__ inline const ActiveCamType& at(const int3& pos) const {
      return buffers_[MemType][posToLinear(pos)];
    }

    template <int MemType = 0>
    __host__ __device__ inline ActiveCamType& at(const int3& pos) {
      return buffers_[MemType][posToLinear(pos)];
    }

    template <int MemType = 0>
    __host__ __device__ inline const ActiveCamType& at(const uint& index) const {
      return buffers_[MemType][index];
    }

    template <int MemType = 0>
    __host__ __device__ inline ActiveCamType& at(const uint& index) {
      return buffers_[MemType][index];
    }

    __host__ __device__ inline const uint size() const {
      return capacity_;
    }

    template <int MemType = 0>
    __host__ __device__ inline const ActiveCamType* data() const {
      return buffers_[MemType];
    }

    template <int MemType = 0>
    __host__ __device__ inline ActiveCamType* data() {
      return buffers_[MemType];
    }

    // copy whole device buffer to host, for debugging at the moment
    __host__ inline void toHost() {
      CUDA_CHECK(cudaMemcpy(buffers_[Host], buffers_[Device], sizeof(ActiveCamType) * capacity_, cudaMemcpyDeviceToHost));
    }

    __host__ inline void toDevice() {
      CUDA_CHECK(cudaMemcpy(buffers_[Device], buffers_[Host], sizeof(ActiveCamType) * capacity_, cudaMemcpyHostToDevice));
    }

    ActiveCamType* buffers_[2] = {nullptr, nullptr};
    CameraGrid_* d_instance_;

    // grid stuff host
    Eigen::Vector3f bucket_extents_;
    Eigen::Vector3f grid_dimensions_;
    Eigen::Vector3f min_grid_pos_;
    Eigen::Vector3f max_grid_pos_;

    // grid stuff device
    float3 d_bucket_extents_;
    float3 d_grid_dimensions_;
    float3 d_min_grid_pos_;
    float3 d_max_grid_pos_;

    uint dim_x_;
    uint dim_y_;
    uint dim_z_;
    uint dim_yx_;
    uint capacity_;

    // active viewpoint selection stuff
    float4* sampling_directions_; // x, y, z direction and number of hits
    int num_samples_;
  };

  struct __align__(16) CameraGridMaxVisibility : CameraGrid_<ActiveCamMaxVisibility> {
    using Base = CameraGrid_<ActiveCamMaxVisibility>;
    CameraGridMaxVisibility(const Eigen::Vector3f& bucket_extents,  // camera blocks extents
                            const Eigen::Vector3f& grid_dimensions, // total grid dim along 3-axis
                            const Eigen::Vector3f& min_grid_pos,    // minimum grid position
                            const Eigen::Vector3f& max_grid_pos,
                            const uint num_samples          = 0,
                            const Camera* camera            = nullptr,
                            const std::string profiler_name = "cam_grid_profiler") :
      CameraGrid_<ActiveCamMaxVisibility>(bucket_extents, grid_dimensions, min_grid_pos, max_grid_pos, num_samples, camera) {
      CUDAProfiler profiler(profiler_name);
      {
        CUDAProfiler::CUDAEvent event(profiler);
        if (camera) {
          // initialize active cams with appropriate buffer
          buffers_[Host] = new ActiveCamMaxVisibility[capacity_]; // construct with default
          // TODO don't like this, for loop to much, do it in CUDA directly
          for (int i = 0; i < capacity_; ++i)
            buffers_[Host][i].allocDevice(camera, sampling_directions_, num_samples_);
        } else {
          // copy stuff to device if allocation use default constructions
          buffers_[Host] = new ActiveCamMaxVisibility[capacity_]; // construct with default
        }
        CUDA_CHECK(cudaMalloc((void**) &buffers_[Device], sizeof(ActiveCamMaxVisibility) * capacity_));
        CUDA_CHECK(cudaMemcpy(buffers_[Device],
                              buffers_[Host],
                              sizeof(ActiveCamMaxVisibility) * capacity_,
                              cudaMemcpyHostToDevice)); // copy from default c++ constructor
      }
      profiler.print(capacity_ * num_samples_);
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(CameraGrid_), cudaMemcpyHostToDevice));
    }

    CameraGridMaxVisibility() = default;

    void calculateBestDirections(const Camera& camera, const std::vector<Sparse>& sparse_container);
    void sortBestDirections(const Camera& camera, const std::vector<Sparse>& sparse_container);
    void calculateVoxelGrid();
  };

  // base version used for data driven
  struct __align__(16) CameraGrid : CameraGrid_<ActiveCam> {
    CameraGrid(const Eigen::Vector3f& bucket_extents,  // camera blocks extents
               const Eigen::Vector3f& grid_dimensions, // total grid dim along 3-axis
               const Eigen::Vector3f& min_grid_pos,    // minimum grid position
               const Eigen::Vector3f& max_grid_pos,
               const uint num_samples = 0,
               const Camera* camera   = nullptr) :
      CameraGrid_<ActiveCam>(bucket_extents, grid_dimensions, min_grid_pos, max_grid_pos, num_samples, camera){};
  };

} // namespace ds