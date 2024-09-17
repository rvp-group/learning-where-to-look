#pragma once
#include "cuda_algebra.cuh"
#include "cuda_matrix.cuh"
#include <Eigen/Core>

namespace ds {
  class Camera {
  public:
    // __host__ Camera() = default;
    __host__ explicit Camera(CUDAMat3 cam_K, const uint rows, const uint cols, const float min_depth, const float max_depth) {
      // copy this ptr to device
      CUDA_CHECK(cudaMalloc((void**) &d_instance_, sizeof(Camera)));
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(Camera), cudaMemcpyHostToDevice));
      // setup some fixed fields
      fx_        = cam_K.row0.x;
      fy_        = cam_K.row1.y;
      ifx_       = 1.f / fx_;
      ify_       = 1.f / fy_;
      cx_        = cam_K.row0.z;
      cy_        = cam_K.row1.z;
      rows_      = rows;
      cols_      = cols;
      min_depth_ = min_depth;
      max_depth_ = max_depth;

      half_h_fov_ = atan2((float) cols_ / 2.f, fx_);
      half_v_fov_ = atan2((float) rows_ / 2.f, fy_);

      threads_ = dim3(N_THREADS_CAM, N_THREADS_CAM);
      blocks_  = dim3((cols_ + N_THREADS_CAM - 1) / N_THREADS_CAM, (rows_ + N_THREADS_CAM - 1) / N_THREADS_CAM);
      // update device ptr
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(Camera), cudaMemcpyHostToDevice));
    }

    __host__ __device__ const void debug() const {
      printf("K:\n%f %f %f %f\nT:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n===========================\n",
             fx_,
             fy_,
             cx_,
             cy_,
             cam_in_world_.rotation.row0.x,
             cam_in_world_.rotation.row0.y,
             cam_in_world_.rotation.row0.z,
             cam_in_world_.translation.x,
             cam_in_world_.rotation.row1.x,
             cam_in_world_.rotation.row1.y,
             cam_in_world_.rotation.row1.z,
             cam_in_world_.translation.y,
             cam_in_world_.rotation.row2.x,
             cam_in_world_.rotation.row2.y,
             cam_in_world_.rotation.row2.z,
             cam_in_world_.translation.z,
             0.f,
             0.f,
             0.f,
             1.f);
    }

    // clang-format off
    const Camera* deviceInstance() const { return d_instance_; }
    const dim3& blocks() const { return blocks_; }
    const dim3& threads() const { return threads_; }

    const uint& cols() const { return cols_; }
    const uint& rows() const { return rows_; }
    const float halfHFoV() const { return half_h_fov_; }
    const float halfVFoV() const { return half_v_fov_; }
    
    
    void computeCloud(CUDAMatrixf3& point_cloud_img);
    void setDepthImage(CUDAMatrixf& depth_img){ depth_img_ = depth_img; }
    void setCamInWorld(const Eigen::Matrix4f& cam_in_world){ cam_in_world_ = CUDAMatSE3(cam_in_world);  CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(Camera), cudaMemcpyHostToDevice));}
    __device__ const float normalizeDepth(const float depth) const;
    __device__ const bool isInCameraFrustumApprox(const float3& pw) const;
    __host__ __device__ const float& maxDepth() const { return max_depth_; }
    __device__ const float& minDepth() const { return min_depth_; }
    __host__ __device__ bool projectPoint(const float3& pw, int2& pimg) const; // device 
    __host__ bool projectPoint(const Eigen::Vector3f& pc, Eigen::Vector2i& pimg) const; // eigen host
    __host__ __device__ float3 inverseProjection(const uint& row,
                                        const uint& col,
                                        const float d) const;

    // just for testing
    // void projectCloud(const CUDAMatrixf3& point_cloud, CUDAMatrixf& depth_img);


    // TODO accessor not working with CUDA 11.8
    // __host__ __device__ const CUDAMatSE3 camInWorld() const { return cam_in_world_; }
    CUDAMatSE3 cam_in_world_;
    // clang-format on

  protected:
    Camera* d_instance_ = nullptr;
    CUDAMatrixf depth_img_;
    dim3 blocks_, threads_;
    uint rows_, cols_;
    float fx_, fy_, ifx_, ify_; // focal lenghts and inverse
    float cx_, cy_;             // principal point
    // horizontal and vertical fov respectively
    float half_h_fov_, half_v_fov_;
    float max_depth_, min_depth_;
  };
} // namespace ds