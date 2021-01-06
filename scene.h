#ifndef CUDA_RAY_TRACER_SCENE_H
#define CUDA_RAY_TRACER_SCENE_H

#include <vector>
#include "surface.h"

struct Object
{
    Coef surface;
    glm::vec3 color;
};

struct Scene
{
    int px_width, px_height;
    float vertical_fov;
    glm::vec3 bg_color;

    std::vector<Object> objects;

    Scene(int px_width, int px_height, float vertical_fov_deg, const glm::vec3& bg_color = glm::vec3(0.0f));
    float aspect_ratio() const { return (float) px_width / (float) px_height; }
};

#endif //CUDA_RAY_TRACER_SCENE_H
