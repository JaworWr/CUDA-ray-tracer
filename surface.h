#ifndef CUDA_RAY_TRACER_SURFACE_H
#define CUDA_RAY_TRACER_SURFACE_H

#include <cuda.h>
#include <glm/glm.hpp>

/*
 * Coefficients of a polynomial with a degree of at most 3
 * For convienicence the degree 3 and 1 coefficients are stored in vectors, eg. the coefficient by x^2y is x2.y
 */
struct Coef
{
    float x3, y3, z3, x2y, xy2, x2z, xz2, y2z, yz2, xyz,
        x2, y2, z2, xy, xz, yz,
        x, y, z, c;
};

/*
 * A surface defined by F(x, y, z) = 0
 * Where F is a polynomial of degree at most 3
 */
struct Surface
{
    Coef coef;

    /*
     * Ray-surface intersection
     * If possible, returns the smallest t >= 0 such that origin + t * dir lies on the surface
     * Otherwise returns something < 0
     */
    __host__ __device__ float intersect_ray(const glm::vec3& origin, const glm::vec3& dir) const;
    __host__ __device__ glm::vec3 normal_vector(const glm::vec3& pos) const;

    // Some nice surfaces
    static Surface sphere(const glm::vec3& center, float radius);
};

#endif //CUDA_RAY_TRACER_SURFACE_H
