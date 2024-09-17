#include "camera_grid.cuh"

namespace ds {

  __global__ void calculateBestDirectionsKernel(const Camera* camera,
                                                const Sparse* landmarks,
                                                const uint landmarks_size,
                                                CameraGridMaxVisibility* active_cam) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    const int3 pos = make_int3(idx, idy, idz);
    // printf("%i %i %i \t ", pos.x, pos.y, pos.z);
    // inside buffer
    if (!active_cam->inside(pos))
      return;
    // printf("%i %i %i \t ", pos.x, pos.y, pos.z);

    // retrieve camera bucket from grid and orientations from grid
    ActiveCamMaxVisibility& cam_bucket = active_cam->at<1>(pos);
    float4* dir_array                  = cam_bucket.d_directions;
    // calculate tranlation and populate isometry for easy projection
    cam_bucket.cam_pos = active_cam->bufferPosToBucket(pos);

    // inside grid - this will automatically set false validity to cam bucket
    if (!active_cam->isInsideGrid(cam_bucket.cam_pos))
      return;

    // printf("| (%i %i %i) (%i %i %i) (%f %f %f) | \t ",
    //        active_cam->dim_x_,
    //        active_cam->dim_y_,
    //        active_cam->dim_z_,
    //        pos.x,
    //        pos.y,
    //        pos.z,
    //        cam_bucket.cam_pos.x,
    //        cam_bucket.cam_pos.y,
    //        cam_bucket.cam_pos.z);

    CUDAMatSE3 cam_in_world;
    cam_in_world.translation = cam_bucket.cam_pos;
    // for each direction check the best suited to observe the landmarks
    for (int i = 0; i < active_cam->num_samples_; ++i) {
      float4& cam_quaternion = cam_bucket.d_directions[i];
      cam_in_world.rotation.fromQuaternion(cam_quaternion);
      // #pragma unroll
      for (int j = 0; j < landmarks_size; ++j) {
        // get point in screen
        const float3 pcam = cam_in_world.inverse() * landmarks[j].pos;

        int2 img_point;
        bool is_good = camera->projectPoint(pcam, img_point);

        if (!is_good)
          continue;

        // easy "z-buffer", just check if already occupied
        const bool& is_occupied = cam_bucket.d_occupied_mat.at<1>(img_point.x, img_point.y);
        if (is_occupied)
          continue;

        // if it is the first time we access the pixel increment count of good directions
        cam_bucket.d_hits_directions[i]++;

        // set the value to occupied
        cam_bucket.d_occupied_mat.at<1>(img_point.x, img_point.y) = 1;
        // and copy ptr of occupied value inside companion array for fast cleanup and increment num
        // note that this is good since we assume a sparse world cam_bucket.occupied_size << cam_bucket.d_occupied_mat.size()
        // if the world is dense this does not make any sense
        cam_bucket.d_occupied_array[cam_bucket.occupied_size] =
          (bool*) (&cam_bucket.d_occupied_mat.at<1>(img_point.x, img_point.y));
        cam_bucket.occupied_size++;
      }

      for (int z = 0; z < cam_bucket.occupied_size; ++z) {
        *cam_bucket.d_occupied_array[z] = 0; // reset value
      }

      cam_bucket.occupied_size = 0;
    }

    for (int i = 0; i < active_cam->num_samples_; ++i) {
      if (cam_bucket.d_hits_directions[i] > cam_bucket.hits_num) {
        cam_bucket.cam_orientation = cam_bucket.d_directions[i];
        cam_bucket.hits_num        = cam_bucket.d_hits_directions[i];
        cam_bucket.valid           = true;
      }
    }
  }

  // ! maximize only the visibility
  void CameraGridMaxVisibility::calculateBestDirections(const Camera& camera, const std::vector<Sparse>& sparse_container) {
    // copy landmarks/world points to device and delete after we get out from scope
    Sparse* d_sparse_container = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_sparse_container, sizeof(Sparse) * sparse_container.size()));
    CUDA_CHECK(
      cudaMemcpy(d_sparse_container, sparse_container.data(), sizeof(Sparse) * sparse_container.size(), cudaMemcpyHostToDevice));

    dim3 threads(N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID);
    dim3 blocks((dim_x_ + threads.x) / threads.x, (dim_y_ + threads.y) / threads.y, (dim_z_ + threads.z) / threads.z);

    // std::cerr << blocks

    calculateBestDirectionsKernel<<<blocks, threads>>>(
      camera.deviceInstance(), d_sparse_container, sparse_container.size(), (CameraGridMaxVisibility*) d_instance_);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_sparse_container));
    // exit(0);
  }

  // ! bubble sort here is fine since the array is small
  __device__ void sort(uint array[], float4 dir_array[], int size) {
    for (int step = 0; step < size - 1; ++step) {
      for (int i = 0; i < size - step - 1; ++i) {
        if (array[i] < array[i + 1]) {
          // swap elements if they are in the wrong order
          uint temp    = array[i];
          array[i]     = array[i + 1];
          array[i + 1] = temp;
          // do the same for direction array
          float4 temp_dir  = dir_array[i];
          dir_array[i]     = dir_array[i + 1];
          dir_array[i + 1] = temp_dir;
        }
      }
    }
  }

  // ! find the n-directions that maximize the visibility
  __global__ void sortBestDirectionsKernel(const Camera* camera,
                                           const Sparse* landmarks,
                                           const uint landmarks_size,
                                           CameraGridMaxVisibility* active_cam) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    const int3 pos = make_int3(idx, idy, idz);
    // inside buffer
    if (!active_cam->inside(pos))
      return;

    // retrieve camera bucket from grid and orientations from grid
    ActiveCamMaxVisibility& cam_bucket = active_cam->at<1>(pos);

    // inside grid - this will automatically set false validity to cam bucket
    if (!active_cam->isInsideGrid(cam_bucket.cam_pos))
      return;

    sort(cam_bucket.d_hits_directions, cam_bucket.d_directions, active_cam->num_samples_);

    // for each direction check the best suited to observe the landmarks
    // if (pos.x == 1 && pos.y == 1 && pos.z == 1) {
    //   printf("sorted vector\n");
    //   for (int i = 0; i < active_cam->num_samples_; ++i) {
    //     printf("num: %d - dir %f %f %f %f | ",
    //            cam_bucket.d_hits_directions[i],
    //            cam_bucket.d_directions[i].x,
    //            cam_bucket.d_directions[i].y,
    //            cam_bucket.d_directions[i].z,
    //            cam_bucket.d_directions[i].w);
    //   }
    // }
  }

  void CameraGridMaxVisibility::sortBestDirections(const Camera& camera, const std::vector<Sparse>& sparse_container) {
    // copy landmarks/world points to device and delete after we get out from scope
    Sparse* d_sparse_container = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_sparse_container, sizeof(Sparse) * sparse_container.size()));
    CUDA_CHECK(
      cudaMemcpy(d_sparse_container, sparse_container.data(), sizeof(Sparse) * sparse_container.size(), cudaMemcpyHostToDevice));

    dim3 threads(N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID);
    // dim3 blocks((dim_x_ + threads.x - 1) / threads.x, (dim_y_ + threads.y - 1) / threads.y, (dim_z_ + threads.z - 1) /
    // threads.z);
    dim3 blocks((dim_x_ + threads.x) / threads.x, (dim_y_ + threads.y) / threads.y, (dim_z_ + threads.z) / threads.z);

    sortBestDirectionsKernel<<<blocks, threads>>>(
      camera.deviceInstance(), d_sparse_container, sparse_container.size(), (CameraGridMaxVisibility*) d_instance_);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_sparse_container));
  }

  // ! calculate only voxel grid (camera poses), not directions/orientations
  // ! method calculate calculateBestDirections already incorporates this!
  __global__ void calculateVoxelGridKernel(CameraGridMaxVisibility* active_cam) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    const int3 pos = make_int3(idx, idy, idz);
    // inside buffer
    if (!active_cam->inside(pos))
      return;

    // retrieve camera bucket from grid and orientations from grid
    ActiveCamMaxVisibility& cam_bucket = active_cam->at<1>(pos);
    // calculate tranlation and populate isometry for easy projection
    cam_bucket.cam_pos = active_cam->bufferPosToBucket(pos);

    // inside grid - this will automatically set false validity to cam bucket
    if (!active_cam->isInsideGrid(cam_bucket.cam_pos))
      return;

    cam_bucket.valid = true;
  }

  void CameraGridMaxVisibility::calculateVoxelGrid() {
    dim3 threads(N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID, N_THREADS_ACTIVE_GRID);
    // dim3 blocks((dim_x_ + threads.x - 1) / threads.x, (dim_y_ + threads.y - 1) / threads.y, (dim_z_ + threads.z - 1) /
    // threads.z);
    dim3 blocks((dim_x_ + threads.x) / threads.x, (dim_y_ + threads.y) / threads.y, (dim_z_ + threads.z) / threads.z);

    calculateVoxelGridKernel<<<blocks, threads>>>((CameraGridMaxVisibility*) d_instance_);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

} // namespace ds