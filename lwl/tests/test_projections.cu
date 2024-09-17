#include <data_structure/camera.cuh>
#include <data_structure/cuda_matrix.cu>
#include <data_structure/cuda_matrix_conversion.cuh>
#include <data_structure/cuda_utils.cuh>
#include <iostream>
#include <vector>

#include "test_utils.cuh"
using namespace ds;

/////////////////////////////////
/////////////// SOME CAMERA UTILS
/////////////////////////////////

__global__ void transformCloudKernel(const CUDAMatrixf3* in_point_cloud, const CUDAMatSE3 T, CUDAMatrixf3* out_point_cloud) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (!in_point_cloud->inside(row, col))
    return;

  const float3& p                  = in_point_cloud->at<1>(row, col);
  out_point_cloud->at<1>(row, col) = T * p;
}

__forceinline__ void transformCloud(const CUDAMatrixf3& in_point_cloud, const CUDAMatSE3& T, CUDAMatrixf3& out_point_cloud) {
  out_point_cloud = CUDAMatrixf3(in_point_cloud.rows(), in_point_cloud.cols());
  out_point_cloud.fill(make_float3(0.f, 0.f, 0.f), true); // fill only in device
  dim3 threads = dim3(N_THREADS_CAM, N_THREADS_CAM);
  dim3 blocks  = dim3((in_point_cloud.cols() + N_THREADS_CAM - 1) / N_THREADS_CAM,
                     (in_point_cloud.rows() + N_THREADS_CAM - 1) / N_THREADS_CAM);
  transformCloudKernel<<<blocks, threads>>>(in_point_cloud.deviceInstance(), T, out_point_cloud.deviceInstance());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// /////////////////////////////////
// /////////////////////////////////
// /////////////////////////////////

constexpr float default_depth = 4.f;

TEST(INV_PROJECTION, FixedDepth) {
  srand(time(NULL));
  // image specs
  uint rows = 480;
  uint cols = 640;
  CUDAMatrixf depth_img(rows, cols);
  depth_img.fill(default_depth);
  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 517.3f, 0.f, 318.6f, 0.f, 516.5f, 255.3f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  // Eigen::Matrix4f cam_in_world = Eigen::Matrix4f::Identity();

  Camera camera(d_cam_K, rows, cols, 0.f, 5.f);
  camera.setDepthImage(depth_img);
  // camera.setCamInWorld(cam_in_world);

  CUDAMatrixf3 point_cloud_img;
  camera.computeCloud(point_cloud_img);
  point_cloud_img.toHost();
  // check that depth value correspond to z of 3d point
  for (uint r = 0; r < point_cloud_img.rows(); ++r) {
    for (uint c = 0; c < point_cloud_img.cols(); ++c) {
      ASSERT_FLOAT_EQ(point_cloud_img.at(r, c).z, default_depth);
    }
  }
}

constexpr float min_depth = 0.f;
constexpr float max_depth = 10.f;

TEST(INV_PROJECTION, RandomDepth) {
  srand(time(NULL));
  // image specs
  uint rows = 480;
  uint cols = 640;
  CUDAMatrixf depth_img(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      depth_img.at(r, c) = generateRandomDepth(min_depth, max_depth);
    }
  }
  depth_img.toDevice();

  // some redundancy just to test everything
  Eigen::Matrix3f cam_K;
  cam_K << 517.3f, 0.f, 318.6f, 0.f, 516.5f, 255.3f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Eigen::Matrix4f cam_in_world = Eigen::Matrix4f::Identity();

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);
  camera.setDepthImage(depth_img);
  // camera.setCamInWorld(cam_in_world);

  CUDAMatrixf3 point_cloud_img, point_cloud_transformed;
  camera.computeCloud(point_cloud_img);
  transformCloud(point_cloud_img, Eig2CUDA(cam_in_world), point_cloud_transformed);
  point_cloud_transformed.toHost();
  // check that depth value correspond to z of 3d point
  for (uint r = 0; r < point_cloud_transformed.rows(); ++r) {
    for (uint c = 0; c < point_cloud_transformed.cols(); ++c) {
      ASSERT_FLOAT_EQ(point_cloud_transformed.at(r, c).z, depth_img.at(r, c));
    }
  }
}

TEST(PROJECTIONS, Dummy) {
  uint rows = 480;
  uint cols = 640;

  Eigen::Matrix3f cam_K;
  cam_K << 300.f, 0.f, 320.f, 0.f, 300.f, 240.f, 0.f, 0.f, 1.f;
  CUDAMat3 d_cam_K(cam_K);

  Camera camera(d_cam_K, rows, cols, min_depth, max_depth);

  for (int i = 0; i < 100000; ++i) {
    Eigen::Vector3f eigen_pcam = Eigen::Vector3f::Random();
    const float3 pcam          = Eig2CUDA(eigen_pcam);

    int2 pimg;
    const bool is_good = camera.projectPoint(pcam, pimg);
    if (!is_good)
      continue;

    // this is the correct order
    const int row = pimg.x;
    const int col = pimg.y;

    float3 pc_reconstructed = camera.inverseProjection(row, col, pcam.z);

    Eigen::Vector3f eigen_pc_reconstructed = CUDA2Eig(pc_reconstructed);
    Eigen::Vector3f error                  = (eigen_pc_reconstructed - eigen_pcam).cwiseAbs();

    ASSERT_LT(error(0), 1e-2); // roundoff due casting
    ASSERT_LT(error(1), 1e-2);
    ASSERT_LT(error(2), 1e-2);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
