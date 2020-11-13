#ifndef CUTA_REDUCE_CUH
#define CUTA_REDUCE_CUH

namespace cuta {
namespace reduce {

/**
 * @brief A CUDA kernel that sums up an array.
 * @tparam T int, float, or double.
 * @param in A pointer to input on device memory.
 * @param out A pointer to output on device memory.
 * @param count The number of elements.
 */
template <typename T>
__global__ void sum(T *out_dev, T *in_dev, unsigned int count);

}  // namespace reduce
}  // namespace cuta

#endif  // CUTA_REDUCE_CUH
