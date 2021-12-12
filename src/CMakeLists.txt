include_directories(${CUDA_ray_tracer_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-10.2/include)

add_library(ray-tracer-common OBJECT
        light.cpp
        scene.cpp
        scene-exception.cpp
        shader-program.cpp
        surface.cpp
        )

add_library(common-libs INTERFACE)
target_link_libraries(common-libs INTERFACE GLEW GL glfw yaml-cpp)

add_library(cuda-libs INTERFACE)
target_link_libraries(cuda-libs INTERFACE cudart)

add_executable(ray-tracer-cpu ray-tracer.cpp update-cpu.cpp)
add_executable(ray-tracer-cuda ray-tracer.cpp update-cuda.cu)

target_link_libraries(ray-tracer-cpu PUBLIC ray-tracer-common common-libs)
target_link_libraries(ray-tracer-cuda PUBLIC ray-tracer-common common-libs)

