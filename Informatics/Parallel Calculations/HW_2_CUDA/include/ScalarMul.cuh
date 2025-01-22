#pragma once

__global__ void ScalarMulBlock(unsigned size, const float* lhs, const float* rhs, float* result);
