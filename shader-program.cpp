#include <GL/glew.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>

#include "shader-program.h"

std::string read_file(const char *path)
{
    std::stringstream ss;
    std::ifstream file(path);
    ss << file.rdbuf();
    return ss.str();
}


ShaderProgram::ShaderProgram(const char *vertex_path, const char *fragment_path)
{
    int success;
    char info_log[512];

    // compile the vertex shader
    fprintf(stderr, "Compiling vertex shader...\n");
    auto vertex_source = read_file(vertex_path);
    auto vertex_source_ptr = vertex_source.c_str();
    unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_source_ptr, nullptr);
    glCompileShader(vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (success != GL_TRUE) {
        glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
        fprintf(stderr, "Vertex shader compilation failed\n%s\n", info_log);
    }

    // compile the fragment shader
    fprintf(stderr, "Compiling fragment shader...\n");
    auto fragment_source = read_file(fragment_path);
    auto fragment_source_ptr = fragment_source.c_str();
    unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_source_ptr, nullptr);
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (success != GL_TRUE) {
        glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
        fprintf(stderr, "Fragment shader compilation failed\n%s\n", info_log);
    }

    // create and link the program
    fprintf(stderr, "Linking...\n");
    m_program_id = glCreateProgram();
    glAttachShader(m_program_id, vertex_shader);
    glAttachShader(m_program_id, fragment_shader);
    glLinkProgram(m_program_id);
    glGetProgramiv(m_program_id, GL_LINK_STATUS, &success);
    if (success != GL_TRUE) {
        glGetProgramInfoLog(m_program_id, 512, nullptr, info_log);
        fprintf(stderr, "Linking failed\n%s\n", info_log);
    }

    // cleanup
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

void ShaderProgram::use() const
{
    glUseProgram(m_program_id);
}
