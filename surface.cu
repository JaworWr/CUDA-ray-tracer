#include "surface.h"

const float TWO_THIRD_PI = M_PI * 2.0f / 3.0f;

__host__ __device__ float Surface::intersect_ray(const glm::vec3 &origin, const glm::vec3 &dir) const
{
    // some helper macros for calculating coefficients
    // the easy coefficients
#define COEF_3(x, y, z) (dir.x * dir.y * dir.z)
#define COEF_0_3(x, y, z) (origin.x * origin.y * origin.z)
#define COEF_2(x, y) (dir.x * dir.y)
#define COEF_0_2(x, y) (origin.x * origin.y)
    // from expansion of (x_0+tx)^3
#define COEF_2_3(x) (3.0f * origin.x * dir.x * dir.x)
#define COEF_1_3(x) (3.0f * origin.x * origin.x * dir.x)
    // from expansion of (x_0+tx)^2(y_0+ty)
#define COEF_2_21(x, y) (dir.x * (dir.x * origin.y + 2.0f * origin.x * dir.y))
#define COEF_1_21(x, y) (origin.x * (origin.x * dir.y + 2.0f * dir.x * origin.y))
    // from expansion of (x_0+tx)(y_0+ty)(z_0+tz)
#define COEF_2_111(x, y, z) (dir.x * dir.y * origin.z + dir.x * origin.y * dir.z + origin.x * dir.y * dir.z)
#define COEF_1_111(x, y, z) (dir.x * origin.y * origin.z + origin.x * dir.y * origin.z + origin.x * origin.y * dir.z)
    // from expansion of (x_0+tx)^2
#define COEF_1_2(x) (2.0f * origin.x * dir.x)
    // from expansion of (x_0+tx)(y_0+ty)
#define COEF_1_11(x, y) (origin.x * dir.y + dir.x * origin.y)

    // coefficients of the polynomial
    float t3 = coef.x3 * COEF_3(x, x, x)
            + coef.y3 * COEF_3(y, y, y)
            + coef.z3 * COEF_3(z, z, z)
            + coef.x2y * COEF_3(x, x, y)
            + coef.xy2 * COEF_3(x, y, y)
            + coef.x2z * COEF_3(x, x, z)
            + coef.xz2 * COEF_3(x, z, z)
            + coef.y2z * COEF_3(y, y, z)
            + coef.yz2 * COEF_3(y, z, z)
            + coef.xyz * COEF_3(x, y, z);
    float t2 = coef.x3 * COEF_2_3(x)
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
    float t1 = coef.x3 * COEF_1_3(x)
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
    float t0 = coef.x3 * COEF_0_3(x, x, x)
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
    if (t3 != 0.0f) {
        // degree = 3
        t2 /= t3;
        t1 /= t3;
        t0 /= t3;
        float q = (3.0f*t1 - t2*t2) / 9.0f;
        float r = (9.0f*t2*t1 - 27.0f*t0 - 2.0f*t2*t2*t2) / 54.0f;
        float q3 = q*q*q;
        float delta = q3 + r*r;
        if (delta > 0.0f) {
            // only one real root - use Cardano's formula
            delta = sqrt(delta);
            q = cbrt(r + delta);
            r = cbrt(r - delta);
            return q + r - t2 / 3.0f;
        }
        else {
            // three real roots - use the trigonometric formula
            float theta = acos(r / sqrt(-q3)) / 3.0f;
            float c = 2.0f * sqrt(-q);
            float x = c * cos(theta) - 1.0f / 3.0f;
            float x1 = c * cos(theta + TWO_THIRD_PI) - 1.0f / 3.0f;
            if (x1 >= 0.0f && x1 < x) {
                x = x1;
            }
            x1 = c * cos(theta + 2.0f * TWO_THIRD_PI) - 1.0f / 3.0f;
            if (x1 >= 0.0f && x1 < x) {
                x = x1;
            }
            return x;
        }

    }
    else if (t2 != 0.0f) {
        // degree = 2
        float delta = t1 * t1 - 4.0f * t2 * t0;
        if (delta < 0.0f) {
            // no solutions
            return -1.0f;
        }
        delta = sqrt(delta);
        float x = (-t1 - delta) / (2.0f * t2);
        if (x >= 0.0f) return x;
        return (-t1 + delta) / (2.0f * t2);
    }
    else if (t1 != 0.0f) {
        // degree = 1
        return -t0 / t1;
    }
    return -1.0f;
}

__host__ __device__ glm::vec3 Surface::normal_vector(const glm::vec3 &pos) const
{
    glm::vec3 res = 3.0f * glm::vec3(coef.x3, coef.y3, coef.z3) * pos * pos
            + 2.0f * glm::vec3(coef.x2, coef.y2, coef.z2) * pos
            + glm::vec3(coef.x, coef.y, coef.z);
    res.x += 2.0f * pos.x * (coef.x2y * pos.y + coef.x2z * pos.z)
            + pos.y * (coef.xy2 * pos.y + coef.xyz * pos.z + coef.xy)
            + pos.z * (coef.xz2 * pos.z + coef.xz);
    res.y += 2.0f * pos.y * (coef.xy2 * pos.x + coef.y2z * pos.z)
            + pos.x * (coef.x2y * pos.x + coef.xyz * pos.z + coef.xy)
            + pos.z * (coef.yz2 * pos.z + coef.yz);
    res.z += 2.0f * pos.z * (coef.xz2 * pos.x + coef.yz2 * pos.y)
            + pos.x * (coef.x2z * pos.x + coef.xyz * pos.y + coef.xz)
            + pos.y * (coef.y2z * pos.y + coef.yz);
    return glm::normalize(res);
}

Surface Surface::sphere(const glm::vec3 &center, float radius)
{
    Coef coef{};
    coef.x2 = coef.y2 = coef.z2 = 1.0f;
    coef.x = -2.0f * center.x;
    coef.y = -2.0f * center.y;
    coef.z = -2.0f * center.z;
    coef.c = glm::dot(center, center) - radius * radius;
    return { coef };
}

Surface Surface::plane(const glm::vec3 &origin, const glm::vec3 &nv)
{
    Coef coef{};
    coef.x = nv.x;
    coef.y = nv.y;
    coef.z = nv.z;
    coef.c = -glm::dot(origin, nv);
    return { coef };
}

Surface Surface::dingDong()
{
    Coef coef{};
    coef.x2 = coef.y2 = coef.z = 1.0f;
    coef.z3 = -1.0f;
    return { coef };
}

Surface Surface::clebsch()
{
    Coef coef{};
    coef.x3 = coef.y3 = coef.x3 = 81.0f;
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -189.0f;
    coef.xyz = 54.0f;
    coef.xy = coef.yz = coef.xz = 126.0f;
    coef.x2 = coef.y2 = coef.z2 = -9.0f;
    coef.x = coef.y = coef.z = 9.0f;
    coef.c = 1.0f;
    return { coef };
}

Surface Surface::cayley()
{
    Coef coef{};
    coef.x2y = coef.x2z = coef.xy2 = coef.y2z = coef.xz2 = coef.yz2 = -5.0f;
    coef.xy = coef.yz = coef.xz = 2.0f;
    return { coef };
}
