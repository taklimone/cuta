#ifndef CUTA_HELPER_CUH
#define CUTA_HELPER_CUH

#include <cstdlib>
#include <iostream>

#define checkCudaAPIError(exp)                                       \
  do {                                                               \
    cudaError_t result = (exp);                                      \
    if (result) {                                                    \
      std::cerr << __FILE__ << ":" << __LINE__ << ":" << #exp << ":" \
                << cudaGetErrorName(result);                         \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  } while (0)

#define checkCudaLaunchError(exp)                                    \
  do {                                                               \
    (exp);                                                           \
    cudaError_t result = cudaGetLastError();                         \
    if (result) {                                                    \
      std::cerr << __FILE__ << ":" << __LINE__ << ":" << #exp << ":" \
                << cudaGetErrorName(result);                         \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  } while (0)

#endif  // CUTA_HELPER_CUH
