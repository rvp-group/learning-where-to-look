#include "test_utils.cuh"
#include <data_structure/camera_grid.cuh>

using namespace ds;

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

__global__ void computeCameraPoseKernel(CameraGridMaxVisibility* active_cam, CameraGridMaxVisibility* active_cam_bkp) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  int idz = threadIdx.z + blockDim.z * blockIdx.z;

  const int3 pos = make_int3(idx, idy, idz);

  if (!active_cam->inside(pos))
    return;

  // first map
  const float3 bucket = active_cam->bufferPosToBucket(pos);

  // inverse map
  const int3 pos_again = active_cam_bkp->bucketToBufferPos(bucket);

  active_cam->at<1>(pos).cam_pos.x = bucket.x;
  active_cam->at<1>(pos).cam_pos.y = bucket.y;
  active_cam->at<1>(pos).cam_pos.z = bucket.z;

  active_cam_bkp->at<1>(pos_again).cam_pos.x = bucket.x;
  active_cam_bkp->at<1>(pos_again).cam_pos.y = bucket.y;
  active_cam_bkp->at<1>(pos_again).cam_pos.z = bucket.z;
}

TEST(ACTIVE_CAMERA, CameraGridIso) {
  const int n_threads = 8;
  // const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.05f, 0.05f, 0.05f);
  // const Eigen::Vector3i grid_dimensions = Eigen::Vector3i(10, 10, 10);

  const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.7f, 0.9f, 1.3f);
  const Eigen::Vector3f grid_dimensions = Eigen::Vector3f(10, 10, 10);

  const Eigen::Vector3f min_grid_pos =
    -Eigen::Vector3f(grid_dimensions.x() / 2, grid_dimensions.y() / 2, grid_dimensions.z() / 2);
  const Eigen::Vector3f max_grid_pos = -min_grid_pos;
  CameraGridMaxVisibility grid(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);
  CameraGridMaxVisibility grid_bkp(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);

  dim3 threads(n_threads, n_threads, n_threads);
  dim3 blocks((grid.dim_x_ + n_threads - 1) / n_threads,
              (grid.dim_y_ + n_threads - 1) / n_threads,
              (grid.dim_z_ + n_threads - 1) / n_threads);

  computeCameraPoseKernel<<<blocks, threads>>>((CameraGridMaxVisibility*) grid.d_instance_,
                                               (CameraGridMaxVisibility*) grid_bkp.d_instance_);
  CUDA_CHECK(cudaDeviceSynchronize());
  grid.toHost();
  grid_bkp.toHost();

  std::cerr << "total grid size: " << grid.size() << std::endl;
  for (uint i = 0; i < grid.size(); ++i) {
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.x, grid_bkp.at<0>(i).cam_pos.x);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.y, grid_bkp.at<0>(i).cam_pos.y);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.z, grid_bkp.at<0>(i).cam_pos.z);
  }
}

TEST(ACTIVE_CAMERA, CameraGridDiff) {
  const int n_threads = 8;
  // const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.013f, 0.05f, 0.015f);
  // const Eigen::Vector3i grid_dimensions = Eigen::Vector3i(9, 4, 10);
  const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.7f, 0.9f, 1.3f);
  const Eigen::Vector3f grid_dimensions = Eigen::Vector3f(10, 10, 10);

  const Eigen::Vector3f min_grid_pos =
    -Eigen::Vector3f(grid_dimensions.x() / 2, grid_dimensions.y() / 2, grid_dimensions.z() / 2);
  const Eigen::Vector3f max_grid_pos = -min_grid_pos;

  CameraGridMaxVisibility grid(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);
  CameraGridMaxVisibility grid_bkp(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);

  dim3 threads(n_threads, n_threads, n_threads);
  dim3 blocks((grid.dim_x_ + n_threads - 1) / n_threads,
              (grid.dim_y_ + n_threads - 1) / n_threads,
              (grid.dim_z_ + n_threads - 1) / n_threads);

  computeCameraPoseKernel<<<blocks, threads>>>((CameraGridMaxVisibility*) grid.d_instance_,
                                               (CameraGridMaxVisibility*) grid_bkp.d_instance_);
  CUDA_CHECK(cudaDeviceSynchronize());
  grid.toHost();
  grid_bkp.toHost();

  std::cerr << "total grid size: " << grid.size() << std::endl;
  for (uint i = 0; i < grid.size(); ++i) {
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.x, grid_bkp.at<0>(i).cam_pos.x);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.y, grid_bkp.at<0>(i).cam_pos.y);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.z, grid_bkp.at<0>(i).cam_pos.z);
  }
}

TEST(ACTIVE_CAMERA, CameraGridFile) {
  const int n_threads                   = 8;
  const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.2f, 0.2f, 0.2f);
  const Eigen::Vector3f grid_dimensions = Eigen::Vector3f(2, 2, 2);
  const Eigen::Vector3f min_grid_pos =
    -Eigen::Vector3f(grid_dimensions.x() / 2, grid_dimensions.y() / 2, grid_dimensions.z() / 2);
  const Eigen::Vector3f max_grid_pos = -min_grid_pos;

  CameraGridMaxVisibility grid(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);
  CameraGridMaxVisibility grid_bkp(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);

  dim3 threads(n_threads, n_threads, n_threads);
  dim3 blocks((grid.dim_x_ + n_threads - 1) / n_threads,
              (grid.dim_y_ + n_threads - 1) / n_threads,
              (grid.dim_z_ + n_threads - 1) / n_threads);

  computeCameraPoseKernel<<<blocks, threads>>>((CameraGridMaxVisibility*) grid.d_instance_,
                                               (CameraGridMaxVisibility*) grid_bkp.d_instance_);
  CUDA_CHECK(cudaDeviceSynchronize());
  grid.toHost();
  grid_bkp.toHost();

  std::string filename = "./grid.txt";
  std::ofstream outfile(filename);
  std::cerr << "total grid size: " << grid.size() << std::endl;
  for (uint i = 0; i < grid.size(); ++i) {
    outfile << grid.at<0>(i).cam_pos.x << " " << grid.at<0>(i).cam_pos.y << " " << grid.at<0>(i).cam_pos.z << "\n";
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.x, grid_bkp.at<0>(i).cam_pos.x);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.y, grid_bkp.at<0>(i).cam_pos.y);
    ASSERT_FLOAT_EQ(grid.at<0>(i).cam_pos.z, grid_bkp.at<0>(i).cam_pos.z);
  }
  outfile.close();
  std::cerr << "open: " << filename << " for inspection" << std::endl;
}

__global__ void transformationKernel(CameraGridMaxVisibility* grid, const float3 d_wp, int3* d_pos, float3* d_bucket) {
  d_pos[0]    = grid->bucketToBufferPos(d_wp);
  d_bucket[0] = grid->bufferPosToBucket(d_pos[0]);
}

TEST(ACTIVE_CAMERA, CameraGridTransformations1) {
  const int n_threads = 8;
  // const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.013f, 0.05f, 0.015f);
  // const Eigen::Vector3i grid_dimensions = Eigen::Vector3i(9, 4, 10);

  const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.7f, 0.9f, 1.3f);
  const Eigen::Vector3f grid_dimensions = Eigen::Vector3f(10, 10, 10);

  const Eigen::Vector3f min_grid_pos =
    -Eigen::Vector3f(grid_dimensions.x() / 2, grid_dimensions.y() / 2, grid_dimensions.z() / 2);
  const Eigen::Vector3f max_grid_pos = -min_grid_pos;

  CameraGridMaxVisibility grid(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);

  // from two close point in the world get buffer position
  Eigen::Vector3f world_point = Eigen::Vector3f(0.013, 0.05, 0.015);
  Eigen::Vector3i pos         = grid.bucketToBufferPos(world_point);

  Eigen::Vector3f world_point2 = Eigen::Vector3f(0.011, 0.04, 0.012);
  Eigen::Vector3i pos2         = grid.bucketToBufferPos(world_point2);

  ASSERT_EQ(pos.x(), pos2.x());
  ASSERT_EQ(pos.y(), pos2.y());
  ASSERT_EQ(pos.z(), pos2.z());

  // get center of bucket
  const Eigen::Vector3f bucket_pos = grid.bufferPosToBucket(pos);
  const Eigen::Vector3f diff       = (bucket_pos - world_point).cwiseAbs();
  const Eigen::Vector3f max_diff   = bucket_extents * 0.5;
  ASSERT_LT(diff.x(), max_diff.x());
  ASSERT_LT(diff.y(), max_diff.y());
  ASSERT_LT(diff.z(), max_diff.z());

  // compare device transformation with host ones
  int3* d_pos;
  CUDA_CHECK(cudaMalloc((void**) &d_pos, sizeof(int3)));
  float3* d_bucket;
  CUDA_CHECK(cudaMalloc((void**) &d_bucket, sizeof(float3)));

  transformationKernel<<<1, 1>>>((CameraGridMaxVisibility*) grid.d_instance_, Eig2CUDA(world_point), d_pos, d_bucket);
  CUDA_CHECK(cudaDeviceSynchronize());

  int3 h_pos;
  float3 h_bucket;
  CUDA_CHECK(cudaMemcpy(&h_pos, d_pos, sizeof(int3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_bucket, d_bucket, sizeof(float3), cudaMemcpyDeviceToHost));

  ASSERT_EQ(pos.x(), h_pos.x);
  ASSERT_EQ(pos.y(), h_pos.y);
  ASSERT_EQ(pos.z(), h_pos.z);

  ASSERT_LT(bucket_pos.x() - h_bucket.x, 1e-5);
  ASSERT_LT(bucket_pos.y() - h_bucket.y, 1e-5);
  ASSERT_LT(bucket_pos.z() - h_bucket.z, 1e-5);
}

TEST(ACTIVE_CAMERA, CameraGridTransformations2) {
  const int n_threads                   = 8;
  const Eigen::Vector3f bucket_extents  = Eigen::Vector3f(0.5, 2, 5);
  const Eigen::Vector3f grid_dimensions = Eigen::Vector3f(3, 6, 20);
  const Eigen::Vector3f min_grid_pos =
    -Eigen::Vector3f(grid_dimensions.x() / 2, grid_dimensions.y() / 2, grid_dimensions.z() / 2);
  const Eigen::Vector3f max_grid_pos = -min_grid_pos;

  CameraGridMaxVisibility grid(bucket_extents, grid_dimensions, min_grid_pos, min_grid_pos);

  // from two close point in the world get buffer position
  Eigen::Vector3f world_point = Eigen::Vector3f(0.15, 1, 2);
  Eigen::Vector3i pos         = grid.bucketToBufferPos(world_point);

  Eigen::Vector3f world_point2 = Eigen::Vector3f(0.13, 1.22, 1.50);
  Eigen::Vector3i pos2         = grid.bucketToBufferPos(world_point2);

  ASSERT_EQ(pos.x(), pos2.x());
  ASSERT_EQ(pos.y(), pos2.y());
  ASSERT_EQ(pos.z(), pos2.z());

  // get center of bucket
  const Eigen::Vector3f bucket_pos = grid.bufferPosToBucket(pos);
  const Eigen::Vector3f diff       = (bucket_pos - world_point).cwiseAbs();
  const Eigen::Vector3f max_diff   = bucket_extents * 0.5;
  ASSERT_LT(diff.x(), max_diff.x());
  ASSERT_LT(diff.y(), max_diff.y());
  ASSERT_LT(diff.z(), max_diff.z());

  // compare device transformation with host ones
  int3* d_pos;
  CUDA_CHECK(cudaMalloc((void**) &d_pos, sizeof(int3)));
  float3* d_bucket;
  CUDA_CHECK(cudaMalloc((void**) &d_bucket, sizeof(float3)));

  transformationKernel<<<1, 1>>>((CameraGridMaxVisibility*) grid.d_instance_, Eig2CUDA(world_point), d_pos, d_bucket);
  CUDA_CHECK(cudaDeviceSynchronize());

  int3 h_pos;
  float3 h_bucket;
  CUDA_CHECK(cudaMemcpy(&h_pos, d_pos, sizeof(int3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_bucket, d_bucket, sizeof(float3), cudaMemcpyDeviceToHost));

  ASSERT_EQ(pos.x(), h_pos.x);
  ASSERT_EQ(pos.y(), h_pos.y);
  ASSERT_EQ(pos.z(), h_pos.z);

  ASSERT_LT(bucket_pos.x() - h_bucket.x, 1e-5);
  ASSERT_LT(bucket_pos.y() - h_bucket.y, 1e-5);
  ASSERT_LT(bucket_pos.z() - h_bucket.z, 1e-5);
}
