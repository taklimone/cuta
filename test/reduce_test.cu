#include <iostream>
#include <vector>

#include "cuta/helper.cuh"
#include "cuta/reduce.cuh"
#include "gtest/gtest.h"

template <typename T>
T cuta_reduce_sum_test(const T *in, unsigned int count) {
  int threads_per_block = 256;
  int num_of_SM;
  checkCudaAPIError(
      cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, 0));
  int num_of_blocks =
      std::min(32 * num_of_SM, static_cast<int>(count) / (threads_per_block));

  std::cerr << "Allocating device memory...";
  T *in_dev;
  T *ret_dev;
  checkCudaAPIError(cudaMalloc(&in_dev, sizeof(T) * count));
  checkCudaAPIError(cudaMalloc(&ret_dev, sizeof(T)));
  std::cerr << "Done.\n";

  std::cerr << "Copying the array from host to device...";
  checkCudaAPIError(
      cudaMemcpy(in_dev, in, sizeof(T) * count, cudaMemcpyHostToDevice));
  std::cerr << "Done.\n";

  std::cerr << "Running cuta::reduce::sum with " << num_of_blocks
            << " blocks of " << threads_per_block << " threads...";
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  checkCudaLaunchError((cuta::reduce::sum<<<num_of_blocks, threads_per_block>>>(
      ret_dev, in_dev, count)));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, start, end);
  std::cerr << "Done. n=" << count << ". " << time << " ms. "
            << (count * sizeof(T) / time / 1'000'000) << " GB/s.\n";

  std::cerr << "Copying the result from device to host...";
  T ret;
  checkCudaAPIError(
      cudaMemcpy(&ret, ret_dev, sizeof(T), cudaMemcpyDeviceToHost));
  std::cerr << "Done.\n";

  std::cerr << "Deallocating device memory...";
  checkCudaAPIError(cudaFree(ret_dev));
  checkCudaAPIError(cudaFree(in_dev));
  std::cerr << "Done.\n";

  return ret;
}

TEST(ReduceTest, ReduceInt) {
  int n = 256 * 1024 * 1024;
  std::vector<int> in(n, 1);
  EXPECT_EQ(cuta_reduce_sum_test(in.data(), n), n);
}

TEST(ReduceTest, ReduceFloat) {
  int n = 256 * 1024 * 1024;
  std::vector<float> in(n, 1.0);
  EXPECT_FLOAT_EQ(cuta_reduce_sum_test(in.data(), n), static_cast<float>(n));
}

#if __CUDA_ARCH__ >= 600
TEST(ReduceTest, ReduceDouble) {
  int n = 256 * 1024 * 1024;
  std::vector<double> in(n, 1.0);
  EXPECT_DOUBLE_EQ(cuta_reduce_sum_test(in.data(), n), static_cast<double>(n));
}
#endif
