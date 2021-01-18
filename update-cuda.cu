#include <cuda.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <cstdio>
#include <glm/gtc/integer.hpp>
#include "helper_cuda_opengl.h"
#include "update.h"
#include "surface_impl.h"
#include "light_impl.h"

cudaGraphicsResource_t resource;
size_t h_width, h_height;
__constant__ size_t d_width, d_height;
__constant__ double d_aspect_ratio, d_vertical_fov;
__constant__ glm::ivec3 d_bg_color;
Object *d_objects;
__constant__ size_t d_n_objects;
LightSource *d_lights;
__constant__ size_t d_n_lights;

const glm::dvec4 RAY_ORIGIN(0.0, 0.0, 0.0, 1.0);
__constant__ glm::dvec4 d_ray_origin;

cudaEvent_t start, end;

size_t idiv(size_t a, size_t b)
{
    return a % b == 0 ? a / b : a / b + 1;
}

void init_update(unsigned int texture, const Scene &scene)
{
    h_width = scene.px_width;
    h_height = scene.px_height;
    double h_aspect_ratio = scene.aspect_ratio();
    double h_vertical_fov = tan(0.5 * scene.vertical_fov);
    auto h_bg_color = glm::iround(scene.bg_color * 255.0f);

    size_t h_n_objects = scene.objects.size();
    checkCudaErrors(cudaMalloc(&d_objects, h_n_objects * sizeof(Object)));
    checkCudaErrors(cudaMemcpy(d_objects, &scene.objects[0], h_n_objects * sizeof(Object), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_n_objects, &h_n_objects, sizeof(size_t), 0, cudaMemcpyHostToDevice));

    size_t h_n_lights = scene.lights.size();
    checkCudaErrors(cudaMalloc(&d_lights, h_n_lights * sizeof(LightSource)));
    checkCudaErrors(cudaMemcpy(d_lights, &scene.lights[0], h_n_lights * sizeof(LightSource), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_n_lights, &h_n_lights, sizeof(size_t), 0, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpyToSymbol(d_width, &h_width, sizeof(size_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_height, &h_height, sizeof(size_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_aspect_ratio, &h_aspect_ratio, sizeof(double), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_vertical_fov, &h_vertical_fov, sizeof(double), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_bg_color, &h_bg_color, sizeof(glm::ivec3), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_ray_origin, &RAY_ORIGIN, sizeof(glm::dvec4), 0, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
}

__global__ void
update_kernel(Object *objects, LightSource *lights, cudaSurfaceObject_t surfaceObject, glm::dmat4 camera_matrix)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    double ndc_x = (tx + 0.5) / d_width;
    double ndc_y = (ty + 0.5) / d_height;
    double camera_x = (2.0 * ndc_x - 1.0) * d_aspect_ratio * d_vertical_fov;
    double camera_y = (2.0 * ndc_y - 1.0) * d_vertical_fov;
    glm::dvec3 dir(camera_x, camera_y, 1.0);
    glm::dvec3 ray_origin = glm::dvec3(camera_matrix * d_ray_origin);
    dir = glm::normalize(glm::dvec3(camera_matrix * glm::dvec4(dir, 1.0)) - ray_origin);

    int best_idx = -1;
    double best_t = INFINITY;
    for (int i = 0; i < d_n_objects; i++) {
        double t = intersect_ray(objects[i].surface, ray_origin, dir);
        if (t >= EPS && t < 1e6 && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }
    glm::ivec3 output_color;
    if (best_idx >= 0) {
        glm::vec3 result_color(0.0f);
        auto surface_point = ray_origin + best_t * dir;
        auto surface_normal = normal_vector(objects[best_idx].surface, surface_point);
        auto object_color = objects[best_idx].color;
        for (int j = 0; j < d_n_lights; j++) {
            double max_t = 0;
            auto shadow_dir = shadow_ray(lights[j], surface_point, max_t);
            bool in_shadow = false;
            for (int k = 0; k < d_n_objects; k++) {
                double t = intersect_ray(objects[k].surface, surface_point + SHADOW_BIAS * surface_normal,
                                                                 shadow_dir);
                if (t > EPS && t < max_t) {
                    in_shadow = true;
                    break;
                }
            }
            if (!in_shadow) {
                result_color += surface_color(lights[j], surface_point, surface_normal, object_color);
            }
        }
        output_color = glm::iround(glm::min(glm::vec3(1.0f), result_color) * 255.0f);
    } else {
        output_color = d_bg_color;
    }

    uchar4 data = {
            (unsigned char) output_color.r,
            (unsigned char) output_color.g,
            (unsigned char) output_color.b,
            255
    };
    if (tx < d_width && ty < d_height) {
        surf2Dwrite(data, surfaceObject, tx * sizeof(uchar4), ty);
    }
}

float update(const glm::dmat4 &camera_matrix)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(idiv(h_width, 16), idiv(h_height, 16));

    cudaArray_t array;
    checkCudaErrors(cudaGraphicsMapResources(1, &resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array;

    cudaSurfaceObject_t surface_object;
    checkCudaErrors(cudaCreateSurfaceObject(&surface_object, &resource_desc));

    cudaEventRecord(start);
    update_kernel<<<gridSize, blockSize>>>(d_objects, d_lights, surface_object, camera_matrix);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    getLastCudaError("update_kernel error");

    checkCudaErrors(cudaDestroySurfaceObject(surface_object));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));
    checkCudaErrors(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}
