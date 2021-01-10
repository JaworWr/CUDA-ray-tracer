#include "light.h"
#include <glm/gtx/norm.hpp>

LightSource LightSource::directional(float intensity, const glm::dvec3 &dir, const glm::vec3 &color)
{
    LightSource light{};
    light.is_spherical = false;
    light.light_color = intensity * color;
    light.p = -glm::normalize(dir);
    return light;
}

LightSource LightSource::spherical(float intensity, const glm::dvec3 &pos, const glm::vec3 &color)
{
    LightSource light{};
    light.is_spherical = true;
    light.light_color = intensity * color;
    light.p = pos;
    return light;
}

glm::vec3 LightSource::surface_color(const glm::dvec3 &surface_point, const glm::dvec3 &surface_norm,
                                     const glm::vec3 &surface_color) const
{
    glm::dvec3 dir;
    glm::vec3 color;
    if (is_spherical) {
        dir = p - surface_point;
        color = light_color / (4.0f * M_PIf32 * (float) glm::length2(dir));
        dir = glm::normalize(dir);
    }
    else {
        dir = p;
        color = light_color;
    }
    return surface_color / M_PIf32 * color * glm::max(0.0f, (float) glm::dot(surface_norm, dir));
}

__device__ glm::vec3 LightSource::surface_color_cuda(const glm::dvec3 &surface_point, const glm::dvec3 &surface_norm,
                                                     const glm::vec3 &surface_color) const
{
    glm::dvec3 dir;
    glm::vec3 color;
    if (is_spherical) {
        dir = p - surface_point;
        color = light_color / (4.0f * M_PIf32 * (float) glm::length2(dir));
        dir = glm::normalize(dir);
    } else {
        dir = p;
        color = light_color;
    }
    return surface_color / M_PIf32 * color * glm::max(0.0f, (float) glm::dot(surface_norm, dir));
}

glm::vec3 LightSource::shadow_ray(const glm::dvec3 &surface_point, double &max_t) const
{
    if (is_spherical) {
        max_t = 1.0f;
        return p - surface_point;
    }
    else {
        max_t = 1e6;
        return p;
    }
}

__device__ glm::vec3 LightSource::shadow_ray_cuda(const glm::dvec3 &surface_point, double &max_t) const
{
    if (is_spherical) {
        max_t = 1.0f;
        return p - surface_point;
    }
    else {
        max_t = 1e6;
        return p;
    }
}
