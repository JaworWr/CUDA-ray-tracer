#include "light.h"
#include "scene-exception.h"

LightSource LightSource::directional(float intensity, const glm::dvec3 &dir, const glm::vec3 &color)
{
    validate_positive("light intensity", intensity);
    validate_color(color);

    LightSource light{};
    light.is_spherical = false;
    light.light_color = intensity * color;
    light.p = -glm::normalize(dir);
    return light;
}

LightSource LightSource::spherical(float intensity, const glm::dvec3 &pos, const glm::vec3 &color)
{
    validate_positive("light intensity", intensity);
    validate_color(color);

    LightSource light{};
    light.is_spherical = true;
    light.light_color = intensity * color;
    light.p = pos;
    return light;
}
