#ifndef CUDA_RAY_TRACER_SURFACE_H
#define CUDA_RAY_TRACER_SURFACE_H

#include <cuda.h>
#include <glm/glm.hpp>

/*
 * Coefficients of a polynomial with a degree of at most 3
 * For convienicence the degree 3 and 1 coefficients are stored in vectors, eg. the coefficient by x^2y is x2.y
 */
struct SurfaceCoefs
{
    double x3, y3, z3, x2y, xy2, x2z, xz2, y2z, yz2, xyz,
        x2, y2, z2, xy, xz, yz,
        x, y, z, c;

    // example surfaces
    static SurfaceCoefs sphere(const glm::dvec3& center, double radius);
    static SurfaceCoefs plane(const glm::dvec3& origin, const glm::dvec3& nv);
    static SurfaceCoefs dingDong(const glm::dvec3& origin);
    static SurfaceCoefs clebsch();
    static SurfaceCoefs cayley();
};

#endif //CUDA_RAY_TRACER_SURFACE_H
