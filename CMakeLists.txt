cmake_minimum_required(VERSION 3.15)
project(cuta LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
