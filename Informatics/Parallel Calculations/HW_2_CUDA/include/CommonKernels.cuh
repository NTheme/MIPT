#pragma once

#include <Filter.cuh>

template <typename Func, typename... Args>
__device__ void Reduce(unsigned size, float *result, Func calc, Args... input) {
  extern __shared__ float shared_data[];

  unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = blockDim.x * gridDim.x;

  shared_data[threadIdx.x] = 0;
  for (unsigned ind = start; ind < size; ind += step) {
    shared_data[threadIdx.x] += calc(ind, input...);
  }
  __syncthreads();

  for (unsigned power = 1; power < blockDim.x; power *= 2) {
    if (threadIdx.x % (2 * power) == 0 && threadIdx.x + power < blockDim.x) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + power];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = shared_data[0];
  }
}

template <typename... Args>
__device__ void ReduceMul(unsigned size, float *result, Args... input) {
  auto calc = [](unsigned ind, const float *lhs, const float *rhs) {
    return lhs[ind] * rhs[ind];
  };

  Reduce(size, result, calc, input...);
}

template <typename... Args>
__device__ void ReduceIDs(unsigned size, float *result, Args... input) {
  auto calc = [](unsigned ind, const float *lhs) {
    return lhs[ind];
  };

  Reduce(size, result, calc, input...);
}

template <typename... Args>
__global__ void ReduceWrapperMul(unsigned size, float *result, Args... input) {
  ReduceMul(size, result, input...);
}

template <typename... Args>
__global__ void ReduceWrapperIDs(unsigned size, float *result, Args... input) {
  ReduceIDs(size, result, input...);
}
