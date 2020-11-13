add_library(cuta reduce.cu)

target_include_directories(cuta PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(
        cuta
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
