/*--========================================--*\
* Author  : NTheme - All rights reserved
* Created : 13 November 2024, 4:16 AM
* File    : MatrixVector.hpp
* Project : Algorithm
\*--========================================--*/

#pragma once

#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace nt {

[[maybe_unused]] __host__ __device__ static inline unsigned int dim3Len(const dim3& size) {
  return size.x * size.y * size.z;
}

[[maybe_unused]] __host__ __device__ static inline unsigned int dimToInd(const dim3& ind, const dim3& size) {
  return ind.x * size.y * size.z + ind.y * size.z + ind.z;
}

[[maybe_unused]] __host__ __device__ static inline dim3 IndToDim(unsigned int ind, const dim3& size) {
  return {ind / (size.y * size.z), (ind % (size.y * size.z)) / size.z, ind % size.z};
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
class VectorCUDA {
 public:
  VectorCUDA() = default;
  explicit VectorCUDA(const dim3& size, const Type& value = Type{});
  ~VectorCUDA();

  Type& operator[](unsigned int ind);
  Type& operator[](const dim3& ind);
  const Type& operator[](unsigned int ind) const;
  const Type& operator[](const dim3& ind) const;
  Type* data() const;

  [[nodiscard]] dim3 size() const;
  void reshape(const dim3& size);

 private:
  dim3 m_size{};
  Type* m_data{};
};

inline static void cudaErrorParse(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error at line " << line << " in file " << file << ": " << cudaGetErrorString(error);
    throw std::runtime_error(ss.str());
  }
}

#define CUDA_ERROR_PARSE(call) cudaErrorParse((call), __FILE__, __LINE__)

inline static void cudaMemoryInfo() {
  size_t freeMem, totalMem;
  CUDA_ERROR_PARSE(cudaMemGetInfo(&freeMem, &totalMem));
  std::cout << "Free: " << freeMem / (1024 * 1024) << "/" << totalMem / (1024 * 1024) << " MB." << std::endl;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
class VectorGPU {
 public:
  VectorGPU() = default;

  template <typename TypeOther>
  explicit VectorGPU(const VectorCUDA<TypeOther>& other);

  __device__ Type& operator[](unsigned int ind);

  Type get(unsigned int ind) const;

 private:
  dim3 m_size{};
  Type* m_data{};
};

template <typename Type>
  requires std::is_arithmetic_v<Type>
template <typename TypeOther>
VectorGPU<Type>::VectorGPU(const VectorCUDA<TypeOther>& other) : m_size(other.size()) {
  CUDA_ERROR_PARSE(cudaMalloc(&m_data, dim3Len(m_size) * sizeof(Type)));
  if constexpr (std::is_same_v<std::remove_cv_t<Type>, std::remove_cv_t<TypeOther>>) {
    CUDA_ERROR_PARSE(cudaMemcpy(m_data, other.data(), dim3Len(m_size) * sizeof(Type), cudaMemcpyHostToDevice));
  } else {
    for (unsigned int ind = 0; ind < dim3Len(m_size); ++ind) {
      Type val = static_cast<Type>(other[ind]);
      CUDA_ERROR_PARSE(cudaMemcpy(m_data + ind, &val, sizeof(Type), cudaMemcpyHostToDevice));
    }
  }
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
__device__ Type& VectorGPU<Type>::operator[](unsigned int ind) {
  return m_data[ind];
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
Type VectorGPU<Type>::get(unsigned int ind) const {
  auto* val_ptr = new Type[1];
  CUDA_ERROR_PARSE(cudaMemcpy(val_ptr, m_data + ind, sizeof(Type), cudaMemcpyDeviceToHost));
  Type val(*val_ptr);
  delete[] val_ptr;
  return val;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
VectorCUDA<Type>::VectorCUDA(const dim3& size, const Type& value) : m_size(size), m_data(new Type[size.x * size.y * size.z]) {
  for (unsigned int ind = 0; ind < size.x * size.y * size.z; ++ind) {
    m_data[ind] = value;
  }
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
VectorCUDA<Type>::~VectorCUDA() {
  delete[] m_data;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
Type& VectorCUDA<Type>::operator[](unsigned int ind) {
  return m_data[ind];
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
Type& VectorCUDA<Type>::operator[](const dim3& ind) {
  return m_data[dimToInd(ind, m_size)];
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
const Type& VectorCUDA<Type>::operator[](unsigned int ind) const {
  return m_data[ind];
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
const Type& VectorCUDA<Type>::operator[](const dim3& ind) const {
  return m_data[dimToInd(ind, m_size)];
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
Type* VectorCUDA<Type>::data() const {
  return m_data;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
dim3 VectorCUDA<Type>::size() const {
  return m_size;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
void VectorCUDA<Type>::reshape(const dim3& size) {
  if (dim3Len(size) != dim3Len(m_size)) {
    throw std::invalid_argument("Wrong shape!");
  }
  m_size = size;
}

template <typename Func, typename... Args>
__device__ void KernelReduce(dim3 size, auto& result, Func calculate, const Args... args) {
  extern __shared__ float shared_data[];

  dim3 start(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
             blockIdx.z * blockDim.z + threadIdx.z);
  dim3 step(gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);

  for (unsigned int y = start.y; y < size.y; y += step.y) {
    for (unsigned int z = start.z; z < size.z; z += step.z) {
      unsigned int ind = dimToInd(threadIdx, blockDim);

      shared_data[ind] = 0;
      for (unsigned int x = start.x; x < size.x; x += step.x) {
        shared_data[ind] += calculate(size, dim3(x, y, z), args...);
      }
      __syncthreads();

      for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        if (threadIdx.x % (shift << 1) == 0 && threadIdx.x + shift < blockDim.x) {
          shared_data[ind] += shared_data[ind + shift];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        result[dimToInd(dim3(blockIdx.x, y, z), size)] = shared_data[ind];
      }
    }
  }
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelReduceIDs(dim3 size, auto* result, const TypeLhs* lhs, [[maybe_unused]] const TypeRhs* rhs) {
  auto calculate = [] __device__(const dim3& size, const dim3& ind, const TypeLhs* lhs) {
    return lhs[dimToInd(ind, size)];
  };
  KernelReduce(size, result, calculate, lhs);
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelReduceSum(dim3 size, auto* result, const TypeLhs* lhs, [[maybe_unused]] const TypeRhs* rhs) {
  auto calculate = [] __device__(const dim3& size, const dim3& ind, const TypeLhs* lhs, const TypeRhs* rhs) {
    return lhs[dimToInd(ind, size)] + rhs[dimToInd(ind, size)];
  };
  KernelReduce(size, result, calculate, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelReduceMul(dim3 size, auto* result, const TypeLhs* lhs, [[maybe_unused]] const TypeRhs* rhs) {
  auto calculate = [] __device__(const dim3& size, const dim3& ind, const TypeLhs* lhs, const TypeRhs* rhs) {
    return lhs[dimToInd(ind, size)] * rhs[dimToInd(ind, size)];
  };
  KernelReduce(size, result, calculate, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelReduceVMl(dim3 size, auto* result, const TypeLhs* lhs, [[maybe_unused]] const TypeRhs* rhs) {
  auto calculate = [] __device__(const dim3& size, const dim3& ind, const TypeLhs* lhs, const TypeRhs* rhs) {
    return lhs[ind.x + size.x * ind.y] * rhs[ind.x * size.z + ind.z];
  };

  KernelReduce(size, result, calculate, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelSum(const dim3& size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  unsigned int start_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int start_y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_z = blockIdx.z * blockDim.z + threadIdx.z;

  unsigned int step_x = blockDim.x * gridDim.x;
  unsigned int step_y = blockDim.y * gridDim.y;
  unsigned int step_z = blockDim.z * gridDim.z;

  for (unsigned int x = start_x; x < size.x; x += step_x) {
    for (unsigned int y = start_y; y < size.y; y += step_y) {
      for (unsigned int z = start_z; z < size.z; z += step_z) {
        unsigned int ind = dimToInd(dim3(x, y, z), size);
        result[ind] = lhs[ind] + rhs[ind];
      }
    }
  }
}

template <typename TypeLhs, typename TypeRhs>
__device__ void KernelMul(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  unsigned int start_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int start_y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_z = blockIdx.z * blockDim.z + threadIdx.z;

  unsigned int step_x = blockDim.x * gridDim.x;
  unsigned int step_y = blockDim.y * gridDim.y;
  unsigned int step_z = blockDim.z * gridDim.z;

  for (unsigned int x = start_x; x < size.x; x += step_x) {
    for (unsigned int y = start_y; y < size.y; y += step_y) {
      for (unsigned int z = start_z; z < size.z; z += step_z) {
        unsigned int ind = dimToInd(dim3(x, y, z), size);
        result[ind] = lhs[ind] * rhs[ind];
      }
    }
  }
}

template <typename TypeLhs, typename TypeRhs>
__global__ void ReduceIDsWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelReduceIDs(size, result, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__global__ void ReduceVMlWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelReduceVMl(size, result, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__global__ void ReduceSumWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelReduceSum(size, result, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__global__ void ReduceMulWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelReduceMul(size, result, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__global__ void SumWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelSum(size, result, lhs, rhs);
}

template <typename TypeLhs, typename TypeRhs>
__global__ void MulWrapper(dim3 size, auto* result, const TypeLhs* lhs, const TypeRhs* rhs) {
  KernelMul(size, result, lhs, rhs);
}

namespace CUDA {

template <typename Type>
  requires std::is_arithmetic_v<Type>
[[maybe_unused]] VectorCUDA<Type> sum1AxisX(const VectorCUDA<Type>& lhs, dim3 block_len = 256) {
  Type* lhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&lhs_cuda, lhs.size().x * lhs.size().y * lhs.size().z * sizeof(Type)));
  CUDA_ERROR_PARSE(cudaMemcpy(lhs_cuda, lhs.data(), dim3Len(lhs.size()) * sizeof(Type), cudaMemcpyHostToDevice));

  Type* res_cuda;
  dim3 size_inp = lhs.size();
  dim3 size_out((size_inp.x + block_len.x - 1) / block_len.x, size_inp.y, size_inp.z);
  dim3 block_cnt(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  CUDA_ERROR_PARSE(cudaMalloc(&res_cuda, dim3Len(size_out) * sizeof(Type)));
  ReduceIDsWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(Type)>>>(size_inp, res_cuda, lhs_cuda, lhs_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  size_inp = size_out;
  size_out = dim3(1, size_inp.y, size_inp.z);
  block_cnt = dim3(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  ReduceIDsWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(Type)>>>(size_inp, res_cuda, res_cuda, res_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  VectorCUDA<Type> result(size_out);
  CUDA_ERROR_PARSE(cudaMemcpy(result.data(), res_cuda, dim3Len(size_out) * sizeof(Type), cudaMemcpyDeviceToHost));
  result.reshape(dim3(size_out.y, size_out.z));

  cudaFree(res_cuda);
  cudaFree(lhs_cuda);

  return result;
}

template <typename TypeLhs, typename TypeRhs>
[[maybe_unused]] VectorCUDA<typename std::common_type<TypeLhs, TypeRhs>::type> sum2AxisX(const VectorCUDA<TypeLhs>& lhs,
                                                                                         const VectorCUDA<TypeRhs>& rhs,
                                                                                         dim3 block_len = 256) {
  if (lhs.size().x != rhs.size().x || lhs.size().y != rhs.size().y || lhs.size().z != rhs.size().z) {
    throw std::invalid_argument("Wrong shape!");
  }

  using TypeResult = typename std::common_type<TypeLhs, TypeRhs>::type;

  TypeLhs* lhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&lhs_cuda, lhs.size().x * lhs.size().y * lhs.size().z * sizeof(TypeLhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(lhs_cuda, lhs.data(), dim3Len(lhs.size()) * sizeof(TypeLhs), cudaMemcpyHostToDevice));

  TypeRhs* rhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&rhs_cuda, dim3Len(rhs.size()) * sizeof(TypeRhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(rhs_cuda, rhs.data(), dim3Len(rhs.size()) * sizeof(TypeRhs), cudaMemcpyHostToDevice));

  TypeResult* res_cuda;
  dim3 size_inp = lhs.size();
  dim3 size_out((size_inp.x + block_len.x - 1) / block_len.x, size_inp.y, size_inp.z);
  dim3 block_cnt(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  CUDA_ERROR_PARSE(cudaMalloc(&res_cuda, dim3Len(size_out) * sizeof(TypeResult)));
  ReduceSumWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(TypeResult)>>>(size_inp, res_cuda, lhs_cuda, rhs_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  size_inp = size_out;
  size_out = dim3(1, size_inp.y, size_inp.z);
  block_cnt = dim3(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  ReduceIDsWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(TypeResult)>>>(size_inp, res_cuda, res_cuda, res_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  VectorCUDA<TypeResult> result(size_out);
  CUDA_ERROR_PARSE(cudaMemcpy(result.data(), res_cuda, dim3Len(size_out) * sizeof(TypeResult), cudaMemcpyDeviceToHost));
  result.reshape(dim3(size_out.y, size_out.z));

  cudaFree(res_cuda);
  cudaFree(rhs_cuda);
  cudaFree(lhs_cuda);

  return result;
}

template <typename TypeLhs, typename TypeRhs>
[[maybe_unused]] VectorCUDA<typename std::common_type<TypeLhs, TypeRhs>::type> dot(const VectorCUDA<TypeLhs>& lhs,
                                                                                   const VectorCUDA<TypeRhs>& rhs,
                                                                                   dim3 block_len = 256) {
  if (lhs.size().y != rhs.size().x) {
    throw std::invalid_argument("Wrong shape: " + std::to_string(lhs.size().x) + " != " + std::to_string(rhs.size().y));
  }

  using TypeResult = typename std::common_type<TypeLhs, TypeRhs>::type;

  TypeLhs* lhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&lhs_cuda, dim3Len(lhs.size()) * sizeof(TypeLhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(lhs_cuda, lhs.data(), dim3Len(lhs.size()) * sizeof(TypeLhs), cudaMemcpyHostToDevice));

  TypeRhs* rhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&rhs_cuda, dim3Len(rhs.size()) * sizeof(TypeRhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(rhs_cuda, rhs.data(), dim3Len(rhs.size()) * sizeof(TypeRhs), cudaMemcpyHostToDevice));

  TypeResult* res_cuda;
  dim3 size_inp(lhs.size().y, lhs.size().x, rhs.size().y);
  dim3 size_out((size_inp.x + block_len.x - 1) / block_len.x, size_inp.y, size_inp.z);
  dim3 block_cnt(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  CUDA_ERROR_PARSE(cudaMalloc(&res_cuda, dim3Len(size_out) * sizeof(TypeResult)));
  ReduceVMlWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(TypeResult)>>>(size_inp, res_cuda, lhs_cuda, rhs_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  size_inp = size_out;
  size_out = dim3(1, size_inp.y, size_inp.z);
  block_cnt = dim3(size_out.x, (size_out.y + block_len.y - 1) / block_len.y, (size_out.z + block_len.z - 1) / block_len.z);
  ReduceIDsWrapper<<<block_cnt, block_len, dim3Len(block_len) * sizeof(TypeResult)>>>(size_inp, res_cuda, res_cuda, res_cuda);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  VectorCUDA<TypeResult> result(size_out);
  CUDA_ERROR_PARSE(cudaMemcpy(result.data(), res_cuda, dim3Len(size_out) * sizeof(TypeResult), cudaMemcpyDeviceToHost));
  result.reshape(dim3(size_out.y, size_out.z));

  cudaFree(res_cuda);
  cudaFree(rhs_cuda);
  cudaFree(lhs_cuda);

  return result;
}

template <typename TypeLhs, typename TypeRhs>
[[maybe_unused]] VectorCUDA<typename std::common_type<TypeLhs, TypeRhs>::type> sum2(const VectorCUDA<TypeLhs>& lhs,
                                                                                    const VectorCUDA<TypeRhs>& rhs,
                                                                                    dim3 block_len = 256) {
  if (lhs.size().x != rhs.size().x || lhs.size().y != rhs.size().y || lhs.size().z != rhs.size().z) {
    throw std::invalid_argument("Wrong shape!");
  }

  using TypeResult = typename std::common_type<TypeLhs, TypeRhs>::type;

  TypeLhs* lhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&lhs_cuda, lhs.size().x * lhs.size().y * lhs.size().z * sizeof(TypeLhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(lhs_cuda, lhs.data(), dim3Len(lhs.size()) * sizeof(TypeLhs), cudaMemcpyHostToDevice));

  TypeRhs* rhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&rhs_cuda, dim3Len(rhs.size()) * sizeof(TypeRhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(rhs_cuda, rhs.data(), dim3Len(rhs.size()) * sizeof(TypeRhs), cudaMemcpyHostToDevice));

  TypeResult* res_cuda;
  dim3 size_inp = lhs.size();
  dim3 size_out = size_inp;
  dim3 block_cnt((size_out.x + block_len.x - 1) / block_len.x, (size_out.y + block_len.y - 1) / block_len.y,
                 (size_out.z + block_len.z - 1) / block_len.z);
  CUDA_ERROR_PARSE(cudaMalloc(&res_cuda, dim3Len(size_out) * sizeof(TypeResult)));
  SumWrapper<<<block_cnt, block_len>>>(size_inp, res_cuda, lhs_cuda, rhs_cuda);
  fflush(stdout);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  VectorCUDA<TypeResult> result(size_out);
  CUDA_ERROR_PARSE(cudaMemcpy(result.data(), res_cuda, dim3Len(size_out) * sizeof(TypeResult), cudaMemcpyDeviceToHost));

  cudaFree(res_cuda);
  cudaFree(rhs_cuda);
  cudaFree(lhs_cuda);

  return result;
}

template <typename TypeLhs, typename TypeRhs>
[[maybe_unused]] VectorCUDA<typename std::common_type<TypeLhs, TypeRhs>::type> mul2(const VectorCUDA<TypeLhs>& lhs,
                                                                                    const VectorCUDA<TypeRhs>& rhs,
                                                                                    dim3 block_len = 256) {
  if (lhs.size().x != rhs.size().x || lhs.size().y != rhs.size().y || lhs.size().z != rhs.size().z) {
    throw std::invalid_argument("Wrong shape!");
  }

  using TypeResult = typename std::common_type<TypeLhs, TypeRhs>::type;

  TypeLhs* lhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&lhs_cuda, lhs.size().x * lhs.size().y * lhs.size().z * sizeof(TypeLhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(lhs_cuda, lhs.data(), dim3Len(lhs.size()) * sizeof(TypeLhs), cudaMemcpyHostToDevice));

  TypeRhs* rhs_cuda;
  CUDA_ERROR_PARSE(cudaMalloc(&rhs_cuda, dim3Len(rhs.size()) * sizeof(TypeRhs)));
  CUDA_ERROR_PARSE(cudaMemcpy(rhs_cuda, rhs.data(), dim3Len(rhs.size()) * sizeof(TypeRhs), cudaMemcpyHostToDevice));

  TypeResult* res_cuda;
  dim3 size_inp = lhs.size();
  dim3 size_out = size_inp;
  dim3 block_cnt((size_out.x + block_len.x - 1) / block_len.x, (size_out.y + block_len.y - 1) / block_len.y,
                 (size_out.z + block_len.z - 1) / block_len.z);
  CUDA_ERROR_PARSE(cudaMalloc(&res_cuda, dim3Len(size_out) * sizeof(TypeResult)));
  MulWrapper<<<block_cnt, block_len>>>(size_inp, res_cuda, lhs_cuda, rhs_cuda);
  fflush(stdout);
  CUDA_ERROR_PARSE(cudaDeviceSynchronize());

  VectorCUDA<TypeResult> result(size_out);
  CUDA_ERROR_PARSE(cudaMemcpy(result.data(), res_cuda, dim3Len(size_out) * sizeof(TypeResult), cudaMemcpyDeviceToHost));

  cudaFree(res_cuda);
  cudaFree(rhs_cuda);
  cudaFree(lhs_cuda);

  return result;
}

}  // namespace CUDA

template <typename Type>
  requires std::is_arithmetic_v<Type>
std::istream& operator>>(std::istream& inp, VectorCUDA<Type>& vector) {
  for (unsigned int x = 0; x < vector.size().x; ++x) {
    for (unsigned int y = 0; y < vector.size().x; ++y) {
      for (unsigned int z = 0; z < vector.size().x; ++z) {
        inp >> vector[x * vector.size().y * vector.size().z + y * vector.size().z + z];
      }
    }
  }
  return inp;
}

template <typename Type>
  requires std::is_arithmetic_v<Type>
std::ostream& operator<<(std::ostream& out, const VectorCUDA<Type>& vector) {
  for (unsigned int x = 0; x < vector.size().x; ++x) {
    for (unsigned int y = 0; y < vector.size().y; ++y) {
      std::cout << "[";
      for (unsigned int z = 0; z < vector.size().z; ++z) {
        out << vector[x * vector.size().y * vector.size().z + y * vector.size().z + z];
        if (z + 1 < vector.size().z) {
          std::cout << ' ';
        } else {
          std::cout << "]";
        }
      }
      if (y + 1 < vector.size().y) {
        out << " ";
      } else {
        std::cout << "\n";
      }
    }
  }
  return out;
}

}  // namespace nt
