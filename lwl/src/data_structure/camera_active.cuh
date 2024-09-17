#pragma once
#include <iostream>
#include <vector>

#include "cuda_matrix_conversion.cuh"
#include "cuda_utils.cuh"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ds {

  // ! directions are always the same for each bucket
  // ! singleton class to allocate these just once
  class __align__(16) Directions {
  public:
    ~Directions() {
      if (d_directions) {
        CUDA_CHECK(cudaFree(d_directions));
      }
    }

    // ! singleton class should not be clonable
    Directions(Directions & other) = delete;

    // ! singleton class should not be assignable
    void operator=(const Directions&) = delete;

    static Directions* GetInstance(const float4* hdata, const uint n_samples) {
      if (instance_ == nullptr) {
        instance_ = new Directions(hdata, n_samples);
      }
      return instance_;
    }

    float4* d_directions = nullptr;

  protected:
    Directions(const float4* hdata, const uint n_samples) {
      CUDA_CHECK(cudaMalloc((void**) &d_directions, sizeof(float4) * n_samples));
      CUDA_CHECK(cudaMemcpy(d_directions, hdata, sizeof(float4) * n_samples, cudaMemcpyHostToDevice));
    }

    static Directions* instance_;
  };

  // Directions* Directions::instance_ = nullptr;

  struct __align__(16) ActiveCamMaxVisibility {
    ActiveCamMaxVisibility() {
      cam_orientation = make_float4(0.f, 0.f, 0.f, 1.f); // qx, qy, qz, qw
      cam_pos         = make_float3(0.f, 0.f, 0.f);
      hits_num        = 0;
      occupied_size   = 0;
      // n_samples         = 0;
      // mat_size          = 0;
      valid             = false;
      d_hits_directions = nullptr;
      d_occupied_array  = nullptr;
      d_directions      = nullptr; // qx, qy, qz, qw
    }

    __host__ void allocDevice(const Camera* camera, const float4* hdata, const uint n_samples) {
      // n_samples = n;
      // calculate this only once, store for each instance the same mem address
      // d_directions = Directions::GetInstance(hdata, n_samples)->d_directions;

      // keep  independent
      CUDA_CHECK(cudaMalloc((void**) &d_directions, sizeof(float4) * n_samples));
      CUDA_CHECK(cudaMemcpy(d_directions, hdata, sizeof(float4) * n_samples, cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMalloc((void**) &d_hits_directions, sizeof(uint) * n_samples));
      CUDA_CHECK(cudaMemset(d_hits_directions, 0, sizeof(uint)));

      // allocate image for light z-buffering, same occupied pixel
      d_occupied_mat.resize(camera->rows(), camera->cols());
      d_occupied_mat.initialize(false);

      CUDA_CHECK(cudaMalloc((void**) &d_occupied_array, sizeof(bool*) * d_occupied_mat.size()));
      CUDA_CHECK(cudaMemset(d_occupied_array, 0, sizeof(bool*) * d_occupied_mat.size()));
    }

    ~ActiveCamMaxVisibility() {
      if (d_hits_directions)
        CUDA_CHECK(cudaFree(d_hits_directions));
      // std::cerr << "d occupied array dest: " << d_occupied_array << std::endl;
      if (d_occupied_array)
        CUDA_CHECK(cudaFree(d_occupied_array));
      if (d_directions)
        CUDA_CHECK(cudaFree(d_directions));
    }

    // inline const Eigen::Vector4f toEigen() const {
    //   return Eigen::Vector4f(cam_orientation.x, cam_orientation.y, cam_orientation.z, cam_orientation.w);
    // }

    inline const Eigen::Quaternionf toEigenQuaternion() const {
      return Eigen::Quaternionf(cam_orientation.w, cam_orientation.x, cam_orientation.y, cam_orientation.z);
    }

    // inline void freeOccupiedPixelsStuff() {
    //   occupied_size = 0;
    //   // d_occupied_mat.clearDeviceBuffer();
    //   // d_occupied_mat.clearHostBuffer();
    //   // if (d_occupied_array) {
    //   //   CUDA_CHECK(cudaFree(d_occupied_array));
    //   //   d_occupied_array = nullptr;
    //   //   std::cerr << "d occupied array: " << d_occupied_array << std::endl;
    //   // }
    // }

    inline bool keepNBestDirections(const uint N) {
      if (!valid)
        return false;
      // copy only the one we are interested in
      std::vector<float4> best_orientations_from_device(N);
      CUDA_CHECK(cudaMemcpy(best_orientations_from_device.data(), d_directions, sizeof(float4) * N, cudaMemcpyDeviceToHost));

      best_hits_num_from_device.resize(N);
      CUDA_CHECK(cudaMemcpy(best_hits_num_from_device.data(), d_hits_directions, sizeof(uint) * N, cudaMemcpyDeviceToHost));

      eigen_best_orientations.resize(N);
      for (uint i = 0; i < N; ++i) {
        eigen_best_orientations[i] = CUDA2Eig(best_orientations_from_device[i]);
      }
      eigen_cam_pose = CUDA2Eig(cam_pos);
      return true;
    }

    // to manage same occupied pixel
    CUDAMatrixb d_occupied_mat;
    bool** d_occupied_array;
    uint mat_size;
    uint occupied_size;

    // for best viewing directions
    float4 cam_orientation;
    float3 cam_pos;

    float4* d_directions; // allocate just once among all the instances

    uint* d_hits_directions;
    uint hits_num;
    bool valid;

    // to keep N-best directions
    // to bind in python
    std::vector<uint> best_hits_num_from_device;
    std::vector<Eigen::Vector4f> eigen_best_orientations;
    Eigen::Vector3f eigen_cam_pose;
  };

  struct __align__(16) ActiveCam {
    ActiveCam() {
      cam_orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
      cam_pos.setZero();
      valid = false;
    }

    // inline const Eigen::Quaternionf toEigenQuaternion() const {
    //   return Eigen::Quaternionf(cam_orientation.w, cam_orientation.x, cam_orientation.y, cam_orientation.z);
    // }

    // for best viewing directions
    Eigen::Vector4d cam_orientation;
    Eigen::Vector3d cam_pos;
    bool valid;
  };

} // namespace ds