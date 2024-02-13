//
// Created by michal on 18.01.2021.
//

#ifndef CUDA_RAY_TRACER_LIGHT_IMPL_H
#define CUDA_RAY_TRACER_LIGHT_IMPL_H

#include "light.h"
#include <glm/gtx/norm.hpp>

#ifdef __CUDA_ARCH__
    #define HOST_OR_DEVICE __device__
#else
    #define HOST_OR_DEVICE
#endif

HOST_OR_DEVICE glm::vec3 shadow_ray(const LightSource &light, const glm::dvec3 &surface_point, double &max_t)
{
    if (light.is_spherical) {
        max_t = 1.0f;
        return light.p - surface_point;
    }
    else {
        max_t = 1e6;
        return light.p;
    }
}

HOST_OR_DEVICE glm::vec3 surface_color(const LightSource &light, const glm::dvec3 &object_point,
                                       const glm::dvec3 &object_norm,
                                       const glm::vec3 &object_color)
{
    glm::dvec3 dir;
    glm::vec3 color;
    if (light.is_spherical) {
        dir = light.p - object_point;
        color = light.light_color / (4.0f * (float) M_PIf32 * (float) glm::length2(dir));
        dir = glm::normalize(dir);
    } else {
        dir = light.p;
        color = light.light_color;
    }
    return object_color / (float) M_PIf32 * color * glm::max(0.0f, (float) glm::dot(object_norm, dir));
}

HOST_OR_DEVICE inline glm::dvec3 reflect_ray(const glm::dvec3 &dir, const glm::dvec3 &normal)
{
    return dir - 2.0 * glm::dot(dir, normal) * normal;
}

#endif //CUDA_RAY_TRACER_LIGHT_IMPL_H
