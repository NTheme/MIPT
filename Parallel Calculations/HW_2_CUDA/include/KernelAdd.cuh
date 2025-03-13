#pragma once

__global__ void KernelAdd(unsigned size, const float* lhs, const float* rhs, float* result);
