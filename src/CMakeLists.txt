add_library(ray-tracer-common OBJECT
        light.cpp
        scene.cpp
        scene-exception.cpp
        shader-program.cpp
        surface.cpp
        ray-tracer.cpp
        )
target_include_directories(ray-tracer-common PUBLIC
        ${INCLUDE_DIR}
        ${PROJECT_BINARY_DIR}/include
        )
target_link_libraries(ray-tracer-common INTERFACE GLEW GL glfw yaml-cpp)

add_library(ray-tracer-cuda-libs INTERFACE)
target_link_libraries(ray-tracer-cuda-libs INTERFACE cudart)

add_executable(ray-tracer-cpu update-cpu.cpp)
target_link_libraries(ray-tracer-cpu PUBLIC ray-tracer-common)
target_compile_options(ray-tracer-cpu PRIVATE -Wall)

add_executable(ray-tracer-cuda update-cuda.cu)
target_link_libraries(ray-tracer-cuda PUBLIC ray-tracer-common ray-tracer-cuda-libs)
target_compile_options(ray-tracer-cuda PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored
            --ptxas-options=-v,-warn-spills
        >
        )
set_target_properties(ray-tracer-cuda PROPERTIES CUDA_ARCHITECTURES "${RAY_TRACER_CUDA_ARCHITECTURES}")

