include_directories(${INCLUDE_DIR})
include_directories(/usr/local/cuda-10.2/include)
include_directories(${PROJECT_BINARY_DIR}/include)

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
target_link_libraries(ray-tracer-cuda PUBLIC ray-tracer-common common-libs cuda-libs)

target_compile_options(ray-tracer-cpu PRIVATE -Wall)
target_compile_options(ray-tracer-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcudafe="--diag_suppress=2929" --ptxas-options=-v,-warn-spills>
        )

