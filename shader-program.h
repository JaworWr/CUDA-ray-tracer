#ifndef CUDA_RAY_TRACER_SHADER_PROGRAM_H
#define CUDA_RAY_TRACER_SHADER_PROGRAM_H


class ShaderProgram {
public:
    ShaderProgram();
    void init(const char *vertex_path, const char *fragment_path);
    void use() const;
private:
    unsigned int m_program_id;
};


#endif //CUDA_RAY_TRACER_SHADER_PROGRAM_H
