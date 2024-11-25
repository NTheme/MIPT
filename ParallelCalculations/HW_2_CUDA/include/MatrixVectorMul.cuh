#pragma once

__global__ void MatrixVectorMul(unsigned height, unsigned width, const float* lhs, const float* rhs, float* result);
