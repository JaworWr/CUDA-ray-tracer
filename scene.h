#ifndef CUDA_RAY_TRACER_SCENE_H
#define CUDA_RAY_TRACER_SCENE_H

#include <vector>
#include "surface.h"
#include "light.h"

struct Object
{
    SurfaceCoefs surface;
    glm::vec3 color;
};

struct Scene
{
    int px_width, px_height;
    double vertical_fov;
    glm::vec3 bg_color;

    std::vector<Object> objects;
    std::vector<LightSource> lights;

    Scene(int px_width, int px_height, double vertical_fov_deg, const glm::vec3& bg_color = glm::vec3(0.0f));
    double aspect_ratio() const { return (double) px_width / px_height; }
};

#endif //CUDA_RAY_TRACER_SCENE_H
