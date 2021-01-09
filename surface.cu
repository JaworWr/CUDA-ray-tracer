#include "surface.h"


const double TWO_THIRD_PI = M_PI * 2.0 / 3.0;

__host__ __device__ double intersect_ray_shared(const Coef& coef, const glm::dvec3 &origin, const glm::dvec3 &dir)
{
    // some helper macros for calculating coefficients
    // the easy coefficients
#define COEF_3(x, y, z) (dir.x * dir.y * dir.z)
#define COEF_0_3(x, y, z) (origin.x * origin.y * origin.z)
#define COEF_2(x, y) (dir.x * dir.y)
#define COEF_0_2(x, y) (origin.x * origin.y)
    // from expansion of (x_0+tx)^3
#define COEF_2_3(x) (3.0 * origin.x * dir.x * dir.x)
#define COEF_1_3(x) (3.0 * origin.x * origin.x * dir.x)
    // from expansion of (x_0+tx)^2(y_0+ty)
#define COEF_2_21(x, y) (dir.x * (dir.x * origin.y + 2.0 * origin.x * dir.y))
#define COEF_1_21(x, y) (origin.x * (origin.x * dir.y + 2.0 * dir.x * origin.y))
    // from expansion of (x_0+tx)(y_0+ty)(z_0+tz)
#define COEF_2_111(x, y, z) (dir.x * dir.y * origin.z + dir.x * origin.y * dir.z + origin.x * dir.y * dir.z)
#define COEF_1_111(x, y, z) (dir.x * origin.y * origin.z + origin.x * dir.y * origin.z + origin.x * origin.y * dir.z)
    // from expansion of (x_0+tx)^2
#define COEF_1_2(x) (2.0 * origin.x * dir.x)
    // from expansion of (x_0+tx)(y_0+ty)
#define COEF_1_11(x, y) (origin.x * dir.y + dir.x * origin.y)

    // coefficients of the polynomial
    double t3 = coef.x3 * COEF_3(x, x, x)
            + coef.y3 * COEF_3(y, y, y)
            + coef.z3 * COEF_3(z, z, z)
            + coef.x2y * COEF_3(x, x, y)
            + coef.xy2 * COEF_3(x, y, y)
            + coef.x2z * COEF_3(x, x, z)
            + coef.xz2 * COEF_3(x, z, z)
            + coef.y2z * COEF_3(y, y, z)
            + coef.yz2 * COEF_3(y, z, z)
            + coef.xyz * COEF_3(x, y, z);
    double t2 = coef.x3 * COEF_2_3(x)
            + coef.y3 * COEF_2_3(y)
            + coef.z3 * COEF_2_3(z)
            + coef.x2y * COEF_2_21(x, y)
            + coef.xy2 * COEF_2_21(y, x)
            + coef.x2z * COEF_2_21(x, z)
            + coef.xz2 * COEF_2_21(z, x)
            + coef.y2z * COEF_2_21(y, z)
            + coef.yz2 * COEF_2_21(z, y)
            + coef.xyz * COEF_2_111(x, y, z)
            + coef.x2 * COEF_2(x, x)
            + coef.y2 * COEF_2(y, y)
            + coef.z2 * COEF_2(z, z)
            + coef.xy * COEF_2(x, y)
            + coef.xz * COEF_2(x, z)
            + coef.yz * COEF_2(y, z);
    double t1 = coef.x3 * COEF_1_3(x)
            + coef.y3 * COEF_1_3(y)
            + coef.z3 * COEF_1_3(z)
            + coef.x2y * COEF_1_21(x, y)
            + coef.xy2 * COEF_1_21(y, x)
            + coef.x2z * COEF_1_21(x, z)
            + coef.xz2 * COEF_1_21(z, x)
            + coef.y2z * COEF_1_21(y, z)
            + coef.yz2 * COEF_1_21(z, y)
            + coef.xyz * COEF_1_111(x, y, z)
            + coef.x2 * COEF_1_2(x)
            + coef.y2 * COEF_1_2(y)
            + coef.z2 * COEF_1_2(z)
            + coef.xy * COEF_1_11(x, y)
            + coef.xz * COEF_1_11(x, z)
            + coef.yz * COEF_1_11(y, z)
            + coef.x * dir.x + coef.y * dir.y + coef.z * dir.z;
    double t0 = coef.x3 * COEF_0_3(x, x, x)
            + coef.y3 * COEF_0_3(y, y, y)
            + coef.z3 * COEF_0_3(z, z, z)
            + coef.x2y * COEF_0_3(x, x, y)
            + coef.xy2 * COEF_0_3(x, y, y)
            + coef.x2z * COEF_0_3(x, x, z)
            + coef.xz2 * COEF_0_3(x, z, z)
            + coef.y2z * COEF_0_3(y, y, z)
            + coef.yz2 * COEF_0_3(y, z, z)
            + coef.xyz * COEF_0_3(x, y, z)
            + coef.x2 * COEF_0_2(x, x)
            + coef.y2 * COEF_0_2(y, y)
            + coef.z2 * COEF_0_2(z, z)
            + coef.xy * COEF_0_2(x, y)
            + coef.xz * COEF_0_2(x, z)
            + coef.yz * COEF_0_2(y, z)
            + coef.x * origin.x + coef.y * origin.y + coef.z * origin.z + coef.c;

    // find the roots of the polynomial
    if (abs(t3) > EPS) {
        // degree = 3
        t2 /= t3;
        t1 /= t3;
        t0 /= t3;
        double q = (3.0*t1 - t2*t2) / 9.0;
        double r = (9.0*t2*t1 - 27.0*t0 - 2.0*t2*t2*t2) / 54.0;
        double delta = q*q*q + r*r;
        if (delta > 1e-8f) {
            // only one real root - use Cardano's formula
            delta = sqrt(delta);
            q = cbrt(r + delta);
            r = cbrt(r - delta);
            return q + r - t2 / 3.0;
        }
        else {
            // three real roots - use the trigonometric formula
            q = min(q, 0.0);
            double theta = acos(r / sqrt(-q*q*q)) / 3.0;
            double c = 2.0 * sqrt(-q);
            double x = c * cos(theta) - t2 / 3.0;
            double x1 = c * cos(theta + TWO_THIRD_PI) - t2 / 3.0;
            if (x1 >= EPS && x1 < x - EPS) {
                x = x1;
            }
            x1 = c * cos(theta + 2.0 * TWO_THIRD_PI) - t2 / 3.0;
            if (x1 >= EPS && x1 < x - EPS) {
                x = x1;
            }
            return x;
        }

    }
    else if (abs(t2) > EPS) {
        // degree = 2
        double delta = t1 * t1 - 4.0 * t2 * t0;
        if (delta < EPS) {
            // no solutions
            return -1.0;
        }
        delta = sqrt(delta);
        double x = (-t1 - delta) / (2.0 * t2);
        if (x >= EPS) return x;
        return (-t1 + delta) / (2.0 * t2);
    }
    else if (abs(t1) > EPS) {
        // degree = 1
        return -t0 / t1;
    }
    return -1.0;
}

__host__ __device__ glm::dvec3 normal_vector_shared(Coef coef, const glm::dvec3 &pos)
{
    glm::dvec3 res = 3.0 * glm::dvec3(coef.x3, coef.y3, coef.z3) * pos * pos
            + 2.0 * glm::dvec3(coef.x2, coef.y2, coef.z2) * pos
            + glm::dvec3(coef.x, coef.y, coef.z);
    res.x += 2.0 * pos.x * (coef.x2y * pos.y + coef.x2z * pos.z)
            + pos.y * (coef.xy2 * pos.y + coef.xyz * pos.z + coef.xy)
            + pos.z * (coef.xz2 * pos.z + coef.xz);
    res.y += 2.0 * pos.y * (coef.xy2 * pos.x + coef.y2z * pos.z)
            + pos.x * (coef.x2y * pos.x + coef.xyz * pos.z + coef.xy)
            + pos.z * (coef.yz2 * pos.z + coef.yz);
    res.z += 2.0 * pos.z * (coef.xz2 * pos.x + coef.yz2 * pos.y)
            + pos.x * (coef.x2z * pos.x + coef.xyz * pos.y + coef.xz)
            + pos.y * (coef.y2z * pos.y + coef.yz);
    return glm::normalize(res);
}

double intersect_ray(Coef coef, const glm::dvec3 &origin, const glm::dvec3 &dir)
{
    return intersect_ray_shared(coef, origin, dir);
}

__device__ double intersect_ray_cuda(Coef coef, const glm::dvec3 &origin, const glm::dvec3 &dir)
{
    return intersect_ray_shared(coef, origin, dir);
}

glm::dvec3 normal_vector(const Coef &coef, const glm::dvec3 &pos)
{
    return normal_vector_shared(coef, pos);
}

__device__ glm::dvec3 normal_vector_cuda(const Coef &coef, const glm::dvec3 &pos)
{
    return normal_vector_shared(coef, pos);
}

Coef sphere(const glm::dvec3 &center, double radius)
{
    Coef coef{};
    coef.x2 = coef.y2 = coef.z2 = 1.0;
    coef.x = -2.0 * center.x;
    coef.y = -2.0 * center.y;
    coef.z = -2.0 * center.z;
    coef.c = glm::dot(center, center) - radius * radius;
    return coef;
}

Coef plane(const glm::dvec3 &origin, const glm::dvec3 &nv)
{
    Coef coef{};
    coef.x = nv.x;
    coef.y = nv.y;
    coef.z = nv.z;
    coef.c = -glm::dot(origin, nv);
    return coef;
}

Coef dingDong(const glm::dvec3& origin)
{
    Coef coef{};
    coef.x2 = coef.y2 = coef.z3 = 1.0;
    coef.z2 = -1.0 - 3.0 * origin.z;
    coef.x = -2.0 * origin.x;
    coef.y = -2.0 * origin.y;
    coef.z = (2.0 + 3.0 * origin.z) * origin.z;
    coef.c = glm::pow(origin.x, 2)
            + glm::pow(origin.y, 2)
            - glm::pow(origin.z, 2) * (1.0 + origin.z);
    return coef;
}

Coef clebsch()
{
    Coef coef{};
    coef.x3 = coef.y3 = coef.x3 = 81.0;
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -189.0;
    coef.xyz = 54.0;
    coef.xy = coef.yz = coef.xz = 126.0;
    coef.x2 = coef.y2 = coef.z2 = -9.0;
    coef.x = coef.y = coef.z = 9.0;
    coef.c = 1.0;
    return coef;
}

Coef cayley()
{
    Coef coef{};
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -5.0;
    coef.xy = coef.yz = coef.xz = 2.0;
    return {coef};
}
