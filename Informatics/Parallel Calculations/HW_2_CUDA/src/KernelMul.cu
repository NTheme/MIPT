#include <KernelMul.cuh>

__global__ void KernelMul(unsigned size, const float* lhs, const float* rhs, float* result) {
  unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = blockDim.x * gridDim.x;

  for (unsigned x = start; x < size; x += step) {
    result[x] = lhs[x] * rhs[x];
  }
}
