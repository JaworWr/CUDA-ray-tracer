cmake_minimum_required(VERSION 3.15)
project(CUDA_ray_tracer LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CUDA_ray_tracer_SOURCE_DIR})

add_subdirectory(src)
