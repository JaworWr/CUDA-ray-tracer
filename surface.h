#ifndef CUDA_RAY_TRACER_SURFACE_H
#define CUDA_RAY_TRACER_SURFACE_H

#include <cuda.h>
#include <glm/glm.hpp>

const float EPS = 1e-8f;

/*
 * Coefficients of a polynomial with a degree of at most 3
 * For convienicence the degree 3 and 1 coefficients are stored in vectors, eg. the coefficient by x^2y is x2.y
 */
struct SurfaceCoefs
{
    double x3, y3, z3, x2y, xy2, x2z, xz2, y2z, yz2, xyz,
        x2, y2, z2, xy, xz, yz,
        x, y, z, c;

    /*
     * Returns the smallest t > 0 such that origin + t * dir lies on the surface
     * If it doesn't exist, returns something < 0
     */
    double intersect_ray(const glm::dvec3 &origin, const glm::dvec3 &dir) const;
#ifdef __NVCC__
    __device__ double intersect_ray_cuda(const glm::dvec3 &origin, const glm::dvec3 &dir) const;
#endif

    /*
     * Normal vector at a given point
     */
    glm::dvec3 normal_vector(const glm::dvec3 &pos) const;
#ifdef __NVCC__
    __device__ glm::dvec3 normal_vector_cuda(const glm::dvec3 &pos) const;
#endif

    // example surfaces
    static SurfaceCoefs sphere(const glm::dvec3& center, double radius);
    static SurfaceCoefs plane(const glm::dvec3& origin, const glm::dvec3& nv);
    static SurfaceCoefs dingDong(const glm::dvec3& origin);
    static SurfaceCoefs clebsch();
    static SurfaceCoefs cayley();
};

#endif //CUDA_RAY_TRACER_SURFACE_H
