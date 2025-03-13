#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(unsigned height, unsigned width, unsigned pitch, const float* lhs, const float* rhs,
                                float* result) {
  unsigned start_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned start_y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned step_x = blockDim.x * gridDim.x;
  unsigned step_y = blockDim.x * gridDim.x;

  for (unsigned x = start_x; x < height; x += step_x) {
    for (unsigned y = start_y; y < width; y += step_y) {
      result[x * pitch + y] = lhs[x * pitch + y] + rhs[x * pitch + y];
    }
  }
}
