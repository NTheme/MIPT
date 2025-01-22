#pragma once

__global__ void MatrixMul(unsigned heightA, unsigned widthA, unsigned widthB, const float* lhs, const float* rhs,
                          float* result);
