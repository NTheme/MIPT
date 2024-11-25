#pragma once

__global__ void KernelMul(unsigned size, const float* lhs, const float* rhs, float* result);
