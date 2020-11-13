# cuta
A hobby project implementing CUDA kernels.

## Prerequisites
- CUDA Toolkit >= 10.0
- CMake >= 3.15

## Build
Specify the compute capability four your NVIDIA GPU through `CMAKE_CUDA_FLAGS` like the following.
```
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="-gencode arch=compute_70,code=sm_70" ..
cmake --build .
```

## Usage
Include headers such as `cuta/reduce.cuh` and link `libcuta.a` to your object code.

## Test
```
ctest --verbose
```
