add_library(ray-tracer-common OBJECT
        light.cpp
        scene.cpp
        scene-exception.cpp
        shader-program.cpp
        surface.cpp
        )
target_include_directories(ray-tracer-common PUBLIC
        ${INCLUDE_DIR}
        ${PROJECT_BINARY_DIR}/include
        )

add_library(common-libs INTERFACE)
target_link_libraries(common-libs INTERFACE GLEW GL glfw yaml-cpp)

add_library(cuda-libs INTERFACE)
target_link_libraries(cuda-libs INTERFACE cudart)

add_executable(ray-tracer-cpu ray-tracer.cpp update-cpu.cpp)
target_link_libraries(ray-tracer-cpu PUBLIC ray-tracer-common common-libs)
target_compile_options(ray-tracer-cpu PRIVATE -Wall)
target_compile_definitions(ray-tracer-cpu PRIVATE RAY_TRACER_CUDA=0)

add_executable(ray-tracer-cuda ray-tracer.cpp update-cuda.cu)
target_link_libraries(ray-tracer-cuda PUBLIC ray-tracer-common common-libs cuda-libs)
target_compile_options(ray-tracer-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcudafe="--diag_suppress=2929" --ptxas-options=-v,-warn-spills>
        )
target_compile_definitions(ray-tracer-cuda PRIVATE RAY_TRACER_CUDA=1)
set_target_properties(ray-tracer-cuda PROPERTIES CUDA_ARCHITECTURES 50)

