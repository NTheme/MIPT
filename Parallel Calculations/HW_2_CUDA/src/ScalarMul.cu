#include <CommonKernels.cuh>
#include <ScalarMul.cuh>

__global__ void ScalarMulBlock(unsigned size, const float* lhs, const float* rhs, float* result) {
  ReduceMul(size, result, lhs, rhs);
}
