#include "camera.cuh"

using namespace ds;

bool Camera::projectPoint(const Eigen::Vector3f& pc, Eigen::Vector2i& pimg) const {
  if (pc.z() <= min_depth_ || pc.z() > max_depth_)
    return false;

  const int row = (fy_ * pc.y() / pc.z() + cy_) + 0.5f;
  const int col = (fx_ * pc.x() / pc.z() + cx_) + 0.5f;

  // const int row = (int) (frow + (frow >= 0 ? 0.5f : -0.5f));
  // const int col = (int) (fcol + (fcol >= 0 ? 0.5f : -0.5f));

  if (row >= 0 && col >= 0 && row < rows_ && col < cols_) {
    pimg << row, col;
    return true;
  }

  return false;
}