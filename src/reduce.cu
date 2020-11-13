#include "cuta/reduce.cuh"

namespace cuta {

namespace reduce {

namespace {

template <typename T>
__device__ T warpSum(T value) {
  for (int i = warpSize / 2; i >= 1; i /= 2) {
    // Butterfly Reduction (This is beautiful.)
    value += __shfl_xor_sync(0xffffffff, value, i, warpSize);
  }
  return value;
}

}  // namespace

template <typename T>
__global__ void sum(T *out_dev, T *in_dev, unsigned int count) {
  T psum{0};

  // Grid-Stride Loop
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    psum += in_dev[i];
  }

  psum = warpSum(psum);

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(out_dev, psum);
  }
}

}  // namespace reduce

}  // namespace cuta

template __device__ int cuta::reduce::warpSum<int>(int);
template __device__ float cuta::reduce::warpSum<float>(float);
template __device__ double cuta::reduce::warpSum<double>(double);

template __global__ void cuta::reduce::sum<int>(int *, int *, unsigned int);
template __global__ void cuta::reduce::sum<float>(float *, float *,
                                                  unsigned int);
#if __CUDA_ARCH__ >= 600
template __global__ void cuta::reduce::sum<double>(double *, double *,
                                                   unsigned int);
#endif
