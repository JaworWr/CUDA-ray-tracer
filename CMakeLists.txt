cmake_minimum_required(VERSION 3.15)
project(CUDA_ray_tracer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(RAY_TRACER_CUDA_ARCHITECTURES "61;75;86" CACHE STRING "CUDA architectures")

add_subdirectory(src)
add_subdirectory(shaders)
