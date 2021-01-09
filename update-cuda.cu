#include <cuda.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <cstdio>
#include <glm/gtc/integer.hpp>
#include "helper_cuda_opengl.h"
#include "update.h"

cudaGraphicsResource_t resource;
size_t h_width, h_height;
double h_aspect_ratio, h_vertical_fov;
glm::ivec3 h_bg_color;
std::vector<Object> h_objects;
__constant__ size_t d_width, d_height;
__constant__ double d_aspect_ratio, d_vertical_fov;
__constant__ glm::ivec3 d_bg_color;
Object* d_objects;
__constant__ size_t d_n_objects;

const glm::dvec3 RAY_ORIGIN(0.0f);
__constant__ glm::dvec3 d_ray_origin;

cudaEvent_t start, end;

size_t idiv(size_t a, size_t b) {
    return a % b == 0 ? a / b : a / b + 1;
}

void init_update(unsigned int texture, const Scene& scene)
{
    h_width = scene.px_width;
    h_height = scene.px_height;
    h_aspect_ratio = scene.aspect_ratio();
    h_vertical_fov = tan(0.5 * scene.vertical_fov);
    h_bg_color = glm::iround(scene.bg_color * 255.0f);

    h_objects = scene.objects;
    size_t n_objects = h_objects.size();
    checkCudaErrors( cudaMalloc(&d_objects, n_objects * sizeof(Object)) );
    checkCudaErrors( cudaMemcpy(d_objects, &h_objects[0], n_objects * sizeof(Object), cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_n_objects, &n_objects, sizeof(size_t), 0, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMemcpyToSymbol(d_width, &h_width, sizeof(size_t), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_height, &h_height, sizeof(size_t), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_aspect_ratio, &h_aspect_ratio, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_vertical_fov, &h_vertical_fov, sizeof(double), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_bg_color, &h_bg_color, sizeof(glm::ivec3), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_ray_origin, &RAY_ORIGIN, sizeof(glm::dvec3), 0, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );

    checkCudaErrors( cudaEventCreate(&start) );
    checkCudaErrors( cudaEventCreate(&end) );
}

__global__ void update_kernel(Object *d_objects, cudaSurfaceObject_t d_surfaceObject)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    double ndc_x = (tx + 0.5) / d_width;
    double ndc_y = (ty + 0.5) / d_height;
    double camera_x = (2.0 * ndc_x - 1.0) * d_aspect_ratio * d_vertical_fov;
    double camera_y = (1.0 - 2.0 * ndc_y) * d_vertical_fov;
    glm::dvec3 dir(camera_x, camera_y, 1.0);
    dir = glm::normalize(dir);

    int best_idx = -1;
    double best_t = INFINITY;
    for (int i = 0; i < d_n_objects; i++) {
        double t = d_objects[i].surface.intersect_ray_cuda(d_ray_origin, dir);
        if (t >= EPS && t < 1e6 && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }
    glm::ivec3 output_color;
    if (best_idx >= 0) {
        output_color = glm::iround(d_objects[best_idx].color * 255.0f);
    }
    else {
        output_color = d_bg_color;
    }

    uchar4 data = {
            (unsigned char) output_color.r,
            (unsigned char) output_color.g,
            (unsigned char) output_color.b,
            255
    };
    if (tx < d_width && ty < d_height) {
        surf2Dwrite(data, d_surfaceObject, tx * sizeof(uchar4), ty);
    }
}

float update()
{
    dim3 blockSize(16, 16);
    dim3 gridSize(idiv(h_width, 16), idiv(h_height, 16));

    cudaArray_t array;
    checkCudaErrors( cudaGraphicsMapResources(1, &resource) );
    checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0) );

    cudaResourceDesc resource_desc = {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array;

    cudaSurfaceObject_t surface_object;
    checkCudaErrors( cudaCreateSurfaceObject(&surface_object, &resource_desc) );

    cudaEventRecord(start);
    update_kernel<<<gridSize, blockSize>>>(d_objects, surface_object);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    getLastCudaError("update_kernel error");

    checkCudaErrors( cudaDestroySurfaceObject(surface_object) );
    checkCudaErrors( cudaGraphicsUnmapResources(1, &resource) );
    checkCudaErrors( cudaDeviceSynchronize() );
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}
