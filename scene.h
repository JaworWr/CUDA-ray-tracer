#ifndef CUDA_RAY_TRACER_SCENE_H
#define CUDA_RAY_TRACER_SCENE_H

#include <vector>
#include <stdexcept>
#include "surface.h"
#include "light.h"

struct Object
{
    SurfaceCoefs surface;
    double reflection_ratio;
    glm::vec3 color;
};

class SceneLoadException :public std::exception
{
public:
    explicit SceneLoadException(std::string error);
    const char * what() const noexcept override;
private:
    std::string error;
};

struct Scene
{
    int px_width, px_height;
    double vertical_fov;
    glm::vec3 bg_color;
    int max_reflections;

    std::vector<Object> objects;
    std::vector<LightSource> lights;

    Scene() = default;
    Scene(int px_width, int px_height, double vertical_fov_deg, int max_reflections, const glm::vec3& bg_color = glm::vec3(0.0f));
    double aspect_ratio() const { return (double) px_width / px_height; }
    static Scene load_from_file(const char *path);
};

#endif //CUDA_RAY_TRACER_SCENE_H
