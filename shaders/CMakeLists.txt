set(SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)
set(VERTEX_SHADER_PATH ${SHADER_DIR}/ray-tracer.vert)
set(FRAGMENT_SHADER_PATH ${SHADER_DIR}/ray-tracer.frag)

configure_file(shader-paths.h.in ${PROJECT_BINARY_DIR}/include/shader-paths.h)
configure_file(ray-tracer.frag ${SHADER_DIR} COPYONLY)
configure_file(ray-tracer.vert ${SHADER_DIR} COPYONLY)
