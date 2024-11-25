#include <MatrixMul.cuh>

__global__ void MatrixMul(unsigned heightA, unsigned widthA, unsigned widthB, const float* lhs, const float* rhs,
                          float* result) {
  extern __shared__ float shared[];
  unsigned start_x = blockIdx.x;
  unsigned start_y = blockIdx.y;
  unsigned start_z = threadIdx.x * blockDim.y + threadIdx.y;

  unsigned step_x = gridDim.x;
  unsigned step_y = gridDim.y;
  unsigned step_z = blockDim.x * blockDim.y;

  for (unsigned x = start_x; x < heightA; x += step_x) {
    for (unsigned y = start_y; y < widthB; y += step_y) {
      shared[start_z] = 0;
      for (unsigned z = start_z; z < widthA; z += step_z) {
        shared[start_z] += lhs[x * widthA + z] * rhs[z * widthB + y];
      }
      __syncthreads();

      for (unsigned power = 1; power < step_z; power <<= 1) {
        if (start_z % (2 * power) == 0) {
          shared[start_z] += shared[start_z + power];
        }
        __syncthreads();
      }

      if (start_z == 0) {
        result[x * widthB + y] = shared[0];
      }
    }
  }
}
