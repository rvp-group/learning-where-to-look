
#include <iostream>

#include "camera_grid.cuh"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Eigen {
  using Vector5f = Eigen::Matrix<float, 5, 1>;
  using Vector6f = Eigen::Matrix<float, 6, 1>;
} // namespace Eigen

namespace ds {

  ds::Directions* ds::Directions::instance_ = nullptr;

  class ActiveCameraGridWrapper {
  public:
    ActiveCameraGridWrapper() {
      grid_   = nullptr;
      camera_ = nullptr;
      K_.setZero();
      bucket_extents_.setZero();
      min_grid_pos_.setZero();
      max_grid_pos_.setZero();
      max_landmark_error_ = 0;
      num_samples_        = 0;
      rows_               = 0;
      cols_               = 0;
      min_depth_          = 0;
      max_depth_          = 0;
    }

    ~ActiveCameraGridWrapper() {
      delete camera_;
      delete grid_;
    }

    void setCamera(const uint rows,
                   const uint cols,
                   const float min_depth,
                   const float max_depth,
                   const float fx,
                   const float fy,
                   const float cx,
                   const float cy) {
      K_ << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f;
      // K frome igen to cuda mat 3x3
      CUDAMat3 d_cam_K(K_);
      // creating cuda camera for projections
      camera_ = new Camera(d_cam_K, rows, cols, min_depth, max_depth);
    }

    void setNumSamples(const uint num_samples) {
      num_samples_ = num_samples;
    }

    void setBucketExtents(const Eigen::Vector3f& bucket_extents) {
      bucket_extents_ = bucket_extents;
    }

    void setGridMinAndMax(const Eigen::Vector3f& min_grid_pos, const Eigen::Vector3f& max_grid_pos) {
      min_grid_pos_ = min_grid_pos;
      max_grid_pos_ = max_grid_pos;
    }

    void setGridDimension(const Eigen::Vector3f& grid_dim) {
      grid_dimensions_ = grid_dim;
    }

    void setSparseLandmarks(const std::vector<Eigen::Vector5f>& sparse_landmarks) {
      // make this cuda buffer
      sparse_container_.resize(sparse_landmarks.size());
      for (int i = 0; i < sparse_landmarks.size(); ++i) {
        Sparse sparse;
        sparse.pos           = Eig2CUDA((Eigen::Vector3f) sparse_landmarks[i].head(3)); // position
        sparse.error         = sparse_landmarks[i](3);                                  // error from reconstruction
        sparse.idx           = sparse_landmarks[i](4);                                  // original index from reconstruction
        sparse_container_[i] = sparse;
      }
    }

    void compute() {
      grid_ = new CameraGridMaxVisibility(bucket_extents_, grid_dimensions_, min_grid_pos_, max_grid_pos_, num_samples_, camera_);

      std::cout << "grid capacity: " << grid_->capacity_ << std::endl;
      std::cout << "\tcam bucket extents: " << bucket_extents_.transpose() << std::endl;
      std::cout << "\tmin grid position: " << min_grid_pos_.transpose() << std::endl;
      std::cout << "\tmax grid position: " << max_grid_pos_.transpose() << std::endl;

      std::cout << "sparse container populated with: " << sparse_container_.size() << std::endl;

      CUDAProfiler active_grid_calculation_profiler("active_grid_calculation");
      {
        CUDAProfiler::CUDAEvent event(active_grid_calculation_profiler);
        grid_->calculateBestDirections(*camera_, sparse_container_);
      }
      active_grid_calculation_profiler.print(grid_->capacity_);
      std::cout << "active view points for each camera calculated" << std::endl;
      grid_->toHost();
      grid_->sortBestDirections(*camera_, sparse_container_);
    }

    std::vector<Eigen::Vector6f> getPoints2D3D(const Eigen::Vector4f& dir, const Eigen::Vector3f& pose) {
      Eigen::Isometry3f pwc = Eigen::Isometry3f::Identity();
      Eigen::Quaternionf qwc(dir(3), dir(0), dir(1), dir(2));
      pwc.linear()                = qwc.normalized().toRotationMatrix();
      pwc.translation()           = pose;
      const Eigen::Isometry3f pcw = pwc.inverse();

      // output vec
      std::vector<Eigen::Vector6f> points2d3d;
      points2d3d.reserve(sparse_container_.size());
      for (int i = 0; i < sparse_container_.size(); ++i) {
        const Eigen::Vector3f pw = CUDA2Eig(sparse_container_[i].pos);
        const Eigen::Vector3f pc = pcw * pw;
        Eigen::Vector2i pimg;
        bool is_good = camera_->projectPoint(pc, pimg);
        if (!is_good)
          continue;
        Eigen::Vector6f allin;
        allin.head(2)           = pimg.cast<float>();
        allin.block<3, 1>(2, 0) = pw;
        allin.tail(1) << sparse_container_[i].error;
        points2d3d.push_back(allin);
      }
      return points2d3d;
    }

    uint const size() const {
      return grid_->capacity_;
    }

    // return the rounded grid pose
    Eigen::Vector3f getPose(const uint index) {
      return CUDA2Eig(grid_->at(index).cam_pos);
    }

    // TODO now maximizes the visibility
    // another data structure should include the directions from MLP
    std::vector<uint> getBestViewingHits(const uint index, const int num_max_directions) {
      if (index > size())
        throw std::runtime_error("getBestViewingHits|index greater than buffer capacity, aborting");

      ActiveCamMaxVisibility& c = grid_->at(index);
      const bool is_ok          = c.keepNBestDirections(num_max_directions);
      if (is_ok) {
        return c.best_hits_num_from_device;
      }
      return {};
    }

    // TODO now maximizes the visibility
    // another data structure should include the directions from MLP
    std::vector<Eigen::Vector4f> getBestViewingDirections(const uint index, const int num_max_directions) {
      if (index > size())
        throw std::runtime_error("getBestViewingDirections|index greater than buffer capacity, aborting");

      ActiveCamMaxVisibility& c = grid_->at(index);
      const bool is_ok          = c.keepNBestDirections(num_max_directions);
      if (is_ok) {
        // std::cerr << index << " " << num_max_directions << " " << c.eigen_cam_pose.transpose() << "\n";
        // for (int j = 0; j < num_max_directions; ++j) {
        //   std::cerr << c.best_hits_num_from_device[j] << " " << c.eigen_best_orientations[j].transpose() << "\n";
        // }
        return c.eigen_best_orientations;
      }
      return {};
    }

  protected:
    CameraGridMaxVisibility* grid_;
    Camera* camera_;
    Eigen::Matrix3f K_;
    Eigen::Vector3f grid_dimensions_;
    Eigen::Vector3f bucket_extents_;       // how big is each buckets, not isotropic
    Eigen::Vector3f min_grid_pos_;         // min grid position
    Eigen::Vector3f max_grid_pos_;         // max grid position, these two are the grid bounds
    std::vector<Sparse> sparse_container_; // buffer containing map of points 3D
    uint rows_, cols_;                     // img dimensions
    uint num_samples_;                     // num spherical samples for directions discretizations
    double max_landmark_error_;            // maximum landmark error [m] coming from reconstruction
    float min_depth_, max_depth_;          // min and max depth for thresholding projections
  };

} // namespace ds