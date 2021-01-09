#include "light.h"
#include <glm/gtx/norm.hpp>

LightSource LightSource::directional(float intensity, const glm::vec3 &dir, const glm::vec3 &color)
{
    LightSource light{};
    light.is_spherical = false;
    light.color = intensity * color;
    light.p = -glm::normalize(dir);
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

glm::vec3 LightSource::surface_color(const glm::vec3 &surface_point, const glm::vec3 &surface_norm,
                                     const glm::vec3 &surface_color) const
{
    if (is_spherical) {
        return surface_color * color / (4.0f * M_PIf32 * glm::distance2(p, surface_point));
    } else {
        return surface_color / M_PIf32 * color * glm::max(0.0f, glm::dot(surface_norm, p));
    }
}

__device__ glm::vec3 LightSource::surface_color_cuda(const glm::vec3 &surface_point, const glm::vec3 &surface_norm,
                                                     const glm::vec3 &surface_color) const
{
    if (is_spherical) {
        return surface_color * color / (4.0f * M_PIf32 * glm::distance2(p, surface_point));
    } else {
        return surface_color / M_PIf32 * color * glm::max(0.0f, glm::dot(surface_norm, p));
    }
}
