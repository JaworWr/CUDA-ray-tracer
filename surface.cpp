#include "surface.h"
#include "scene-exception.h"

SurfaceCoefs SurfaceCoefs::sphere(const glm::dvec3 &center, double radius)
{
    validate_positive("sphere radius", radius);

    SurfaceCoefs coef{};
    coef.x2 = coef.y2 = coef.z2 = 1.0;
    coef.x = -2.0 * center.x;
    coef.y = -2.0 * center.y;
    coef.z = -2.0 * center.z;
    coef.c = glm::dot(center, center) - radius * radius;
    return coef;
}

SurfaceCoefs SurfaceCoefs::plane(const glm::dvec3 &origin, const glm::dvec3 &nv)
{
    SurfaceCoefs coef{};
    coef.x = nv.x;
    coef.y = nv.y;
    coef.z = nv.z;
    coef.c = -glm::dot(origin, nv);
    return coef;
}

SurfaceCoefs SurfaceCoefs::dingDong(const glm::dvec3& origin)
{
    SurfaceCoefs coef{};
    coef.x2 = coef.y3 = coef.z2 = 1.0;
    coef.y2 = -1.0 - 3.0 * origin.y;
    coef.x = -2.0 * origin.x;
    coef.z = -2.0 * origin.z;
    coef.y = (2.0 + 3.0 * origin.y) * origin.y;
    coef.c = glm::pow(origin.x, 2)
            + glm::pow(origin.z, 2)
            - glm::pow(origin.y, 2) * (1.0 + origin.y);
    return coef;
}

SurfaceCoefs SurfaceCoefs::clebsch()
{
    SurfaceCoefs coef{};
    coef.x3 = coef.y3 = coef.x3 = 81.0;
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -189.0;
    coef.xyz = 54.0;
    coef.xy = coef.yz = coef.xz = 126.0;
    coef.x2 = coef.y2 = coef.z2 = -9.0;
    coef.x = coef.y = coef.z = 9.0;
    coef.c = 1.0;
    return coef;
}

SurfaceCoefs SurfaceCoefs::cayley()
{
    SurfaceCoefs coef{};
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -5.0;
    coef.xy = coef.yz = coef.xz = 2.0;
    return {coef};
}
