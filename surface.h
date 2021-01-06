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
 * Returns the smallest t > 0 such that origin + t * dir lies on the surface
 * If it doesn't exist, returns something < 0
 */
float intersect_ray(const Coef& coef, const glm::vec3 &origin, const glm::vec3 &dir);
/*
 * Normal vector at a given point
 */
glm::vec3 normal_vector(const Coef& coef, const glm::vec3 &pos);

#ifdef __CUDA_ARCH__
__device__ float intersect_ray_cuda(const Coef& coef, const glm::vec3 &origin, const glm::vec3 &dir);
__device__ glm::vec3 normal_vector_cuda(const Coef& coef, const glm::vec3 &pos);
#endif

// example surfaces
Coef sphere(const glm::vec3& center, float radius);
Coef plane(const glm::vec3& origin, const glm::vec3& nv);
Coef dingDong();
Coef clebsch();
Coef cayley();

#endif //CUDA_RAY_TRACER_SURFACE_H
