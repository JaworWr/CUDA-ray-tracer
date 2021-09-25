#ifndef CUDA_RAY_TRACER_SCENE_H
#define CUDA_RAY_TRACER_SCENE_H

#include <vector>
#include "surface.h"
#include "light.h"

struct Object
{
    SurfaceCoefs surface;
    float reflection_ratio;
    glm::vec3 color;

    Object(SurfaceCoefs surface, float reflection_ratio, const glm::vec3 &color);
};

struct Scene
{
    unsigned int px_width, px_height;
    double vertical_fov;
    glm::vec3 bg_color;
    unsigned int max_reflections;

    std::vector<Object> objects;
    std::vector<LightSource> lights;

    Scene() = default;

    Scene(unsigned int px_width, unsigned int px_height, double vertical_fov_deg, unsigned int max_reflections,
          const glm::vec3 &bg_color = glm::vec3(0.0f));

    double aspect_ratio() const
    { return (double) px_width / px_height; }

    static Scene load_from_file(const char *path);
};

#endif //CUDA_RAY_TRACER_SCENE_H
