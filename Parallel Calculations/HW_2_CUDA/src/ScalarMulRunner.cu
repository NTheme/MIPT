#include <CommonKernels.cuh>
#include <KernelMul.cuh>
#include <ScalarMul.cuh>
#include <ScalarMulRunner.cuh>

float ScalarMulTwoReductions(unsigned size, const float* lhs, const float* rhs, unsigned block_size) {
  unsigned block1_cnt = (size + block_size - 1) / block_size;
  unsigned block2_cnt = (block1_cnt + block_size - 1) / block_size;

  float* lhs_cuda;
  float* rhs_cuda;
  float* result1_cuda;
  float* result2_cuda;
  auto* result = new float[block2_cnt];

  cudaMalloc(&lhs_cuda, size * sizeof(float));
  cudaMalloc(&rhs_cuda, size * sizeof(float));
  cudaMalloc(&result1_cuda, size * sizeof(float));
  cudaMalloc(&result2_cuda, block1_cnt * sizeof(float));

  cudaMemcpy(lhs_cuda, lhs, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_cuda, rhs, size * sizeof(float), cudaMemcpyHostToDevice);

  float ans = 0;
  KernelMul<<<block1_cnt, block_size>>>(size, lhs_cuda, rhs_cuda, result2_cuda);
  cudaDeviceSynchronize();
  ReduceWrapperIDs<<<block1_cnt, block_size, block_size * sizeof(float)>>>(size, result1_cuda, result2_cuda);
  cudaDeviceSynchronize();
  ReduceWrapperIDs<<<block2_cnt, block_size, block_size * sizeof(float)>>>(block1_cnt, result2_cuda, result1_cuda);
  cudaDeviceSynchronize();

  cudaMemcpy(result, result2_cuda, block2_cnt * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < block2_cnt; ++i)
    ans += result[i];

  delete[] result;

  cudaFree(lhs_cuda);
  cudaFree(rhs_cuda);
  cudaFree(result1_cuda);
  cudaFree(result2_cuda);

  return ans;
}

float ScalarMulSumPlusReduction(unsigned size, const float* lhs, const float* rhs, unsigned block_size) {
  unsigned block1_cnt = (size + block_size - 1) / block_size;

  float* lhs_cuda;
  float* rhs_cuda;
  float* result_cuda;

  cudaMalloc(&lhs_cuda, size * sizeof(float));
  cudaMalloc(&rhs_cuda, size * sizeof(float));
  cudaMalloc(&result_cuda, block1_cnt * sizeof(float));

  cudaMemcpy(lhs_cuda, lhs, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_cuda, rhs, size * sizeof(float), cudaMemcpyHostToDevice);

  float ans = 0;
  ScalarMulBlock<<<block1_cnt, block_size, block_size * sizeof(float)>>>(size, lhs_cuda, rhs_cuda, result_cuda);
  cudaDeviceSynchronize();
  ReduceWrapperIDs<<<1, block_size, block_size * sizeof(float)>>>(block1_cnt, result_cuda, result_cuda);
  cudaDeviceSynchronize();
  cudaMemcpy(&ans, result_cuda, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(lhs_cuda);
  cudaFree(rhs_cuda);
  cudaFree(result_cuda);

  return ans;
}
