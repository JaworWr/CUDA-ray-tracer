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
size_t h_width, h_height, h_n_objects, h_n_lights;
__constant__ size_t d_width, d_height;
__constant__ double d_aspect_ratio, d_vertical_fov;
__constant__ glm::ivec3 d_bg_color;
Object *d_objects;
__constant__ size_t d_n_objects;
LightSource *d_lights;
__constant__ size_t d_n_lights;

const glm::dvec4 RAY_ORIGIN(0.0, 0.0, 0.0, 1.0);
__constant__ glm::dvec4 d_ray_origin;

glm::dvec3 *d_ray_dirs;
double *d_t_arr;
glm::dvec3 *d_intersection_points;
glm::dvec3 *d_normal_vectors;
glm::vec3 *d_surface_colors;
glm::vec3 *d_pixel_colors;

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
//    auto h_bg_color = glm::iround(scene.bg_color * 255.0f);
    auto h_bg_color = scene.bg_color;

    h_n_objects = scene.objects.size();
    checkCudaErrors(cudaMalloc(&d_objects, h_n_objects * sizeof(Object)));
    checkCudaErrors(cudaMemcpy(d_objects, &scene.objects[0], h_n_objects * sizeof(Object), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_n_objects, &h_n_objects, sizeof(size_t), 0, cudaMemcpyHostToDevice));

    h_n_lights = scene.lights.size();
    checkCudaErrors(cudaMalloc(&d_lights, h_n_lights * sizeof(LightSource)));
    checkCudaErrors(cudaMemcpy(d_lights, &scene.lights[0], h_n_lights * sizeof(LightSource), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_n_lights, &h_n_lights, sizeof(size_t), 0, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpyToSymbol(d_width, &h_width, sizeof(size_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_height, &h_height, sizeof(size_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_aspect_ratio, &h_aspect_ratio, sizeof(double), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_vertical_fov, &h_vertical_fov, sizeof(double), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_bg_color, &h_bg_color, sizeof(glm::ivec3), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_ray_origin, &RAY_ORIGIN, sizeof(glm::dvec4), 0, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&d_ray_dirs, h_width * h_height * sizeof(glm::dvec3)));
    checkCudaErrors(cudaMalloc(&d_t_arr, h_width * h_height * h_n_objects * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_intersection_points, h_width * h_height * sizeof(glm::dvec3)));
    checkCudaErrors(cudaMalloc(&d_normal_vectors, h_width * h_height * sizeof(glm::dvec3)));
    checkCudaErrors(cudaMalloc(&d_surface_colors, h_width * h_height * sizeof(glm::vec3)));
    checkCudaErrors(cudaMalloc(&d_pixel_colors, h_width * h_height * sizeof(glm::vec3)));

    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&end));
}

__global__ void calc_ray(glm::dmat4 camera_matrix, glm::dvec3 *ray_dir)
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

    if (tx < d_width && ty < d_height) {
        ray_dir[ty * d_width + tx] = dir;
    }
}

__global__ void
calc_intersection_t(const Object *objects, glm::dvec3 ray_origin, const glm::dvec3 *ray_dir, double *t_arr)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tz = blockIdx.z * blockDim.z + threadIdx.z;

    if (tx >= d_width || ty >= d_height || tz >= d_n_objects) return;
    auto dir = ray_dir[ty * d_width + tx];
    double t = intersect_ray(objects[tz].surface, ray_origin, dir);
    t_arr[tz * d_width * d_height + ty * d_width + tx] = t;
}

__global__ void
calc_intersection_point(const Object *objects, glm::dvec3 ray_origin, const glm::dvec3 *ray_dir, const double *t_arr,
                        glm::dvec3 *intersection_points, glm::dvec3 *normal_vectors, glm::vec3 *surface_colors)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= d_width || ty >= d_height) return;

    int best_idx = -1;
    double best_t = INFINITY;
    for (int i = 0; i < d_n_objects; i++) {
        double t = t_arr[i * d_width * d_height + ty * d_width + tx];
        if (t >= EPS && t < 1e6 && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }
    auto intersection = ray_origin + best_t * ray_dir[ty * d_width + tx];
    intersection_points[ty * d_width + tx] = intersection;
    intersection_points[ty * d_width + tx] = intersection;
    normal_vectors[ty * d_width + tx] = normal_vector(objects[best_idx].surface, intersection);
    if (best_t < INFINITY) {
        surface_colors[ty * d_width + tx] = objects[best_idx].color;
    }
    else {
        surface_colors[ty * d_width + tx] = glm::vec3(-1.0f);
    }
}

__global__ void calc_light(const Object *objects, const LightSource *lights, const glm::dvec3 *intersection_points,
                           const glm::dvec3 *normal_vectors, const glm::vec3 *surface_colors, glm::vec3 *pixel_colors)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tz = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ glm::vec3 colors[];

    glm::vec3 light(0.0f);
    if (tx >= d_width || ty >= d_height || tz >= d_n_lights) return;
    auto object_color = surface_colors[ty * d_width + tx];
    if (object_color.x >= 0.0f) {
        auto intersection_point = intersection_points[ty * d_width + tx];
        auto normal_vector = normal_vectors[ty * d_width + tx];
        light = surface_color(lights[tz], intersection_point, normal_vector, object_color);
        double max_t = 0;
        auto shadow_dir = shadow_ray(lights[tz], intersection_point, max_t);
        auto shadow_origin = intersection_point + SHADOW_BIAS * normal_vector;
        for (int k = 0; k < d_n_objects; k++) {
            double t = intersect_ray(objects[k].surface, shadow_origin, shadow_dir);
            if (t > EPS && t < max_t) {
                light = glm::vec3(0.0f);
                break;
            }
        }
    }
    else if (tz == 0) {
        pixel_colors[ty * d_width + tx] = d_bg_color;
    }

    colors[threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] = light;
    for (size_t d = blockDim.z >> 1; d >= 1; d >>= 1) {
        __syncthreads();
        if (threadIdx.z + d < blockDim.z) {
            colors[threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x]
                    += colors[(threadIdx.z + d) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x];
        }
    }
    __syncthreads();
    if (threadIdx.z == 0) {
        atomicAdd(&pixel_colors[ty * d_width + tx].x,
                  colors[threadIdx.y * blockDim.x + threadIdx.x].x);
        atomicAdd(&pixel_colors[ty * d_width + tx].y,
                  colors[threadIdx.y * blockDim.x + threadIdx.x].y);
        atomicAdd(&pixel_colors[ty * d_width + tx].z,
                  colors[threadIdx.y * blockDim.x + threadIdx.x].z);
    }
}

__global__ void write_pixels(const glm::vec3 *pixel_colors, cudaSurfaceObject_t surface_object)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= d_width || ty >= d_height) return;
    auto output_color = glm::iround(glm::clamp(pixel_colors[ty * d_width + tx], 0.0f, 1.0f) * 255.0f);
    uchar4 data = {
            (unsigned char) output_color.r,
            (unsigned char) output_color.g,
            (unsigned char) output_color.b,
            255
    };
    surf2Dwrite(data, surface_object, tx * sizeof(uchar4), ty);
}

float update(const glm::dmat4 &camera_matrix)
{
    dim3 blockSize2d(16, 16);
    dim3 gridSize2d(idiv(h_width, 16), idiv(h_height, 16));
    dim3 blockSize3d(4, 4, 16);
    dim3 gridSize3d_objects(idiv(h_width, 4), idiv(h_height, 4), idiv(h_n_objects, 16));
    dim3 gridSize3d_lights(idiv(h_width, 4), idiv(h_height, 4), idiv(h_n_lights, 16));

    cudaArray_t array;
    checkCudaErrors(cudaGraphicsMapResources(1, &resource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array;

    cudaSurfaceObject_t surface_object;
    checkCudaErrors(cudaCreateSurfaceObject(&surface_object, &resource_desc));

    cudaEventRecord(start);
//    update_kernel<<<gridSize2d, blockSize2d>>>(d_objects, d_lights, surface_object, camera_matrix);
//    getLastCudaError("update_kernel error");

    auto ray_origin = camera_matrix * RAY_ORIGIN;
    calc_ray<<<gridSize2d, blockSize2d>>>(camera_matrix, d_ray_dirs);
    getLastCudaError("calc_ray error");
    calc_intersection_t<<<gridSize3d_objects, blockSize3d>>>(d_objects, ray_origin, d_ray_dirs, d_t_arr);
    getLastCudaError("calc_intersection_t error");
    calc_intersection_point<<<gridSize2d, blockSize2d>>>(d_objects, ray_origin, d_ray_dirs, d_t_arr,
                                                         d_intersection_points, d_normal_vectors, d_surface_colors);
    getLastCudaError("calc_intersection_point error");
    checkCudaErrors(cudaMemset(d_pixel_colors, 0, h_width * h_height * sizeof(glm::vec3)));
    calc_light<<<gridSize3d_lights, blockSize3d, 4*4*16*sizeof(glm::vec3)>>>(d_objects, d_lights, d_intersection_points, d_normal_vectors,
                                                   d_surface_colors, d_pixel_colors);
    getLastCudaError("calc_light error");
    cudaDeviceSynchronize();
    write_pixels<<<gridSize2d, blockSize2d>>>(d_pixel_colors, surface_object);
    getLastCudaError("write_pixels error");
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    checkCudaErrors(cudaDestroySurfaceObject(surface_object));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource));
    checkCudaErrors(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}
