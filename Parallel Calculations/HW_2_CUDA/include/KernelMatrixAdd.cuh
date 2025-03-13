#pragma once

__global__ void KernelMatrixAdd(unsigned height, unsigned width, unsigned pitch, const float* lhs, const float* rhs,
                                float* result);
