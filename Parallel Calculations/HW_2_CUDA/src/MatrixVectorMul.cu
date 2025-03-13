#include <MatrixVectorMul.cuh>

__global__ void MatrixVectorMul(unsigned height, unsigned width, const float* lhs, const float* rhs, float* result) {
  unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = blockDim.x * gridDim.x;

  for (unsigned x = start; x < height; x += step) {
    result[x] = 0;
    for (int y = 0; y < width; ++y) {
      result[x] += lhs[x * width + y] * rhs[y];
    }
  }
}
