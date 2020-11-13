configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/googletest-download"
        )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/googletest-download"
        )

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
        "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
        )


add_executable(reduce_test reduce_test.cu)
target_include_directories(reduce_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/include)
target_link_libraries(reduce_test PRIVATE cuta gtest_main)
add_test(NAME reduce_test COMMAND reduce_test)
