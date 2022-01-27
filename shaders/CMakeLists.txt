set(VERTEX_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ray-tracer.vert)
set(FRAGMENT_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ray-tracer.frag)

configure_file(
        shader-paths.h.in
        ${PROJECT_BINARY_DIR}/include/shader-paths.h
)
