#include "camera.cuh"
#include "cuda_matrix.cu"

using namespace ds;

__global__ void calculateCloudKernel(const CUDAMatrixf* depth_img, const Camera* camera, CUDAMatrixf3* point_cloud_img) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (!depth_img->inside(row, col))
    return;

  const float& depth_val = depth_img->at<1>(row, col);

  if (depth_val <= camera->minDepth() || depth_val > camera->maxDepth())
    return;

  float3 pcam                      = camera->inverseProjection(row, col, depth_val);
  point_cloud_img->at<1>(row, col) = pcam;
}

float3 Camera::inverseProjection(const uint& row, const uint& col, const float d) const {
  return d * make_float3(ifx_ * (col - cx_ - 0.5f), ify_ * (row - cy_ - 0.5f), 1.f);
}

const float Camera::normalizeDepth(const float depth) const {
  return (depth - min_depth_) / (max_depth_ - min_depth_);
}

const bool Camera::isInCameraFrustumApprox(const float3& pw) const {
  const float3 pcam = cam_in_world_.inverse() * pw;

  int2 pimg;
  const bool is_good = projectPoint(pcam, pimg);

  if (!is_good)
    return false;
  return true;
  // TODO this is from the original paper
  // float3 pimg3 = make_float3(pimg.x, pimg.y, pcam.z);
  // // some normalization, we want the stuff to be between 0 and 1
  // pimg3.x = (2.f * pimg3.x - (cols_ - 1.f)) / (cols_ - 1.f);
  // pimg3.y = ((rows_ - 1.f) - 2.f * pimg3.y) / (rows_ - 1.f);
  // pimg3.z = (pimg3.z - min_depth_) / (max_depth_ - min_depth_);
  // pimg3 *= 0.95f; // TODO
  // return !(pimg3.x < -1.f || pimg3.x > 1.f || pimg3.y < -1.f || pimg3.y > 1.f || pimg3.z < 0.f || pimg3.z > 1.f);
  // return true;
}

bool Camera::projectPoint(const float3& pc, int2& pimg) const {
  if (pc.z <= min_depth_ || pc.z > max_depth_)
    return false;

  const int row = (fy_ * pc.y / pc.z + cy_) + 0.5f;
  const int col = (fx_ * pc.x / pc.z + cx_) + 0.5f;

  // const int row = (int) (frow + (frow >= 0 ? 0.5f : -0.5f));
  // const int col = (int) (fcol + (fcol >= 0 ? 0.5f : -0.5f));

  if (row >= 0 && col >= 0 && row < rows_ && col < cols_) {
    pimg = make_int2(row, col);
    return true;
  }

  return false;
}

void Camera::computeCloud(CUDAMatrixf3& point_cloud_img) {
  point_cloud_img = CUDAMatrixf3(rows_, cols_);
  point_cloud_img.fill(make_float3(0.f, 0.f, 0.f), true); // fill only in device
  calculateCloudKernel<<<blocks_, threads_>>>(depth_img_.deviceInstance(), d_instance_, point_cloud_img.deviceInstance());
  CUDA_CHECK(cudaDeviceSynchronize());
}

/////////////////////////////////////////////////////////////////////
////////////////////// TESTING, USELESS ////////////////////////////
/////////////////////////////////////////////////////////////////////

// __global__ void projectCloudKernel(const CUDAMatrixf3* point_cloud_img, const Camera* camera, CUDAMatrixf* depth_img) {
//   int row = blockDim.y * blockIdx.y + threadIdx.y;
//   int col = blockDim.x * blockIdx.x + threadIdx.x;

//   if (!point_cloud_img->inside(row, col))
//     return;

//   const float3& pc = point_cloud_img->at<1>(row, col);
//   int2 pimg;
//   camera->projectPoint(pc, pimg);

//   printf("%d %d | %d %d\n", pimg.x, pimg.y, row, col);
//   depth_img->at<1>(pimg.x, pimg.y) = pc.z;
// }

// void Camera::projectCloud(const CUDAMatrixf3& point_cloud, CUDAMatrixf& depth_img) {
//   depth_img = CUDAMatrixf(rows_, cols_);
//   depth_img.fill(0.f, true);
//   projectCloudKernel<<<blocks_, threads_>>>(point_cloud.deviceInstance(), d_instance_, depth_img.deviceInstance());
//   CUDA_CHECK(cudaDeviceSynchronize());
// }