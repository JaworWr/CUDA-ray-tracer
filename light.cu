#include "light.h"
#include <glm/gtx/norm.hpp>

LightSource LightSource::directional(float intensity, const glm::vec3 &dir, const glm::vec3 &color)
{
    LightSource light{};
    light.is_spherical = false;
    light.color = intensity * color;
    light.p = dir;
    return light;
}

LightSource LightSource::spherical(float intensity, const glm::vec3 &pos, const glm::vec3 &color)
{
    LightSource light{};
    light.is_spherical = true;
    light.color = intensity * color;
    light.p = pos;
    return light;
}

__host__ __device__ glm::vec3
surface_color_directional(const glm::vec3 &light_dir, const glm::vec3 &light_color, const glm::vec3 &surface_norm,
                          const glm::vec3 &surface_color)
{
    return surface_color / M_PIf32 * light_color * glm::max(0.0f, glm::dot(surface_norm, light_dir));
}

__host__ __device__ glm::vec3
surface_color_spherical(const glm::vec3 &light_pos, const glm::vec3 &light_color, const glm::vec3 &surface_pos,
                        const glm::vec3 &surface_color)
{
    return surface_color * light_color / (4.0f * M_PIf32 * glm::distance2(light_pos, surface_pos));
}

glm::vec3 LightSource::surface_color(const glm::dvec3 &surface_point, const glm::dvec3 &surface_norm,
                                     const glm::vec3 &surface_color) const
{
    if (is_spherical) {
        return surface_color_spherical(p, color, surface_point, surface_color);
    } else {
        return surface_color_directional(p, color, surface_norm, surface_color);
    }
}

__device__ glm::vec3 LightSource::surface_color_cuda(const glm::dvec3 &surface_point, const glm::dvec3 &surface_norm,
                                                     const glm::vec3 &surface_color) const
{
    if (is_spherical) {
        return surface_color_spherical(p, color, surface_point, surface_color);
    } else {
        return surface_color_directional(p, color, surface_norm, surface_color);
    }
}
