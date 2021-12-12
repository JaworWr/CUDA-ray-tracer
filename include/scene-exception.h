#ifndef CUDA_RAY_TRACER_SCENE_EXCEPTION_H
#define CUDA_RAY_TRACER_SCENE_EXCEPTION_H

#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <glm/glm.hpp>

class SceneException : public std::exception
{
public:
    explicit SceneException(std::string error)
            : error{std::move(error)}
    {}

    const char *what() const noexcept override
    {
        return error.c_str();
    }

private:
    std::string error;
};

template<typename T>
void validate_positive(const char *what, const T &value)
{
    if (value < 0) {
        std::stringstream error;
        error << "Negative value for " << what << ": " << value;
        throw SceneException(error.str());
    }
}

void validate_color(const glm::vec3 &color);

#endif //CUDA_RAY_TRACER_SCENE_EXCEPTION_H
