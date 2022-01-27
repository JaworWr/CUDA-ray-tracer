#ifndef CUDA_RAY_TRACER_LIGHT_H
#define CUDA_RAY_TRACER_LIGHT_H

#include <glm/glm.hpp>

struct LightSource {
    static LightSource directional(float intensity, const glm::dvec3 &dir, const glm::vec3 &color);
    static LightSource spherical(float intensity, const glm::dvec3 &pos, const glm::vec3 &color);

    bool is_spherical;
    glm::dvec3 p;
    glm::vec3 light_color;
};


#endif //CUDA_RAY_TRACER_LIGHT_H
