#pragma once

float ScalarMulTwoReductions(unsigned size, const float* lhs, const float* rhs, unsigned block_size);

float ScalarMulSumPlusReduction(unsigned size, const float* lhs, const float* rhs, unsigned block_size);
