#pragma once

enum OperationFilterType {
  GT,
  LT
};

__global__ void Filter(unsigned size, const float* array, OperationFilterType type, const float* value, float* result,
                       unsigned* pos);