#ifndef CUDA_RAY_TRACER_LIGHT_H
#define CUDA_RAY_TRACER_LIGHT_H

#include <cuda.h>
#include <glm/glm.hpp>

class LightSource
{
public:
    static LightSource directional(float intensity, const glm::dvec3 &dir, const glm::vec3 &color);
    static LightSource spherical(float intensity, const glm::dvec3 &pos, const glm::vec3 &color);

    glm::vec3 surface_color(const glm::dvec3 &surface_point,
                            const glm::dvec3 &surface_norm,
                            const glm::vec3 &surface_color) const;

#ifdef __NVCC__
    __device__ glm::vec3 surface_color_cuda(const glm::dvec3 &surface_point,
                            const glm::dvec3 &surface_norm,
                            const glm::vec3 &surface_color) const;
#endif

    glm::vec3 shadow_ray(const glm::dvec3 &surface_point, double &max_t) const;

#ifdef __NVCC__
    __device__ glm::vec3 shadow_ray_cuda(const glm::dvec3 &surface_point, double &max_t) const;
#endif
private:
    bool is_spherical;
    glm::dvec3 p;
    glm::vec3 color;
};

#endif //CUDA_RAY_TRACER_LIGHT_H
