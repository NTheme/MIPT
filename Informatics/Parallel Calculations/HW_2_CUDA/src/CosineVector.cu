#include <cmath>
#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>

float CosineVector(unsigned size, const float* lhs, const float* rhs, unsigned blockSize) {
  float l1 = std::sqrt(ScalarMulTwoReductions(size, lhs, lhs, blockSize));
  float l2 = std::sqrt(ScalarMulTwoReductions(size, rhs, rhs, blockSize));
  return ScalarMulTwoReductions(size, lhs, rhs, blockSize) / (l1 * l2);
}
