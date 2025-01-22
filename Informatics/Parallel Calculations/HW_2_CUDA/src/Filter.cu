#include <CommonKernels.cuh>
#include <Filter.cuh>

__global__ void Filter(unsigned size, const float* array, OperationFilterType type, const float* value, float* result,
                       unsigned* pos) {
  extern __shared__ float shared[];
  unsigned start = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = blockDim.x * gridDim.x;

  shared[threadIdx.x] = 0;
  for (unsigned i = start; i < size; i += step) {
    bool appropriate = false;
    if (type == GT) {
      appropriate = array[i] > *value;
    } else {
      appropriate = array[i] < *value;
    }

    shared[threadIdx.x] += appropriate ? 1 : 0;
  }
  __syncthreads();

  for (unsigned int power = 1; power < blockDim.x; power *= 2) {
    if (threadIdx.x % (2 * power) == 0 && threadIdx.x + power < blockDim.x) {
      shared[threadIdx.x] += shared[threadIdx.x + power];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    unsigned cur = atomicAdd(pos, static_cast<unsigned int>(shared[0]));
    for (int shift = 0; shift < blockDim.x; ++shift) {
      for (unsigned ind = start + shift; ind < size; ind += step) {
        bool appropriate = false;
        if (type == GT) {
          appropriate = array[ind] > *value;
        } else {
          appropriate = array[ind] < *value;
        }

        if (appropriate) {
          result[cur++] = array[ind];
        }
      }
    }
  }
}
