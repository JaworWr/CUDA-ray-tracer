#include <cuda.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <cstdio>
#include <glm/gtc/integer.hpp>
#include "helper_cuda_opengl.h"
#include "update.h"

cudaGraphicsResource_t resource;
size_t h_width, h_height;
float h_aspect_ratio, h_vertical_fov;
glm::ivec3 h_bg_color;
__constant__ size_t d_width, d_height;
__constant__ float d_aspect_ratio, d_vertical_fov;
__constant__ glm::ivec3 d_bg_color;

size_t idiv(size_t a, size_t b) {
    return a % b == 0 ? a / b : a / b + 1;
}

void init_update(unsigned int texture, const Scene& scene)
{
    h_width = scene.px_width;
    h_height = scene.px_height;
    h_aspect_ratio = scene.aspect_ratio();
    h_vertical_fov = scene.vertical_fov;
    h_bg_color = glm::iround(scene.bg_color * 255.0f);

    checkCudaErrors( cudaMemcpyToSymbol(d_width, &h_width, sizeof(int), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_height, &h_height, sizeof(int), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_aspect_ratio, &h_aspect_ratio, sizeof(float), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_vertical_fov, &h_vertical_fov, sizeof(float), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_bg_color, &h_bg_color, sizeof(glm::ivec3), 0, cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
}

__global__ void update_kernel(cudaSurfaceObject_t d_surfaceObject)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    // inverted - just to see that it's the kernel that's running and not the CPU version
    uchar4 data = {
            (unsigned char) (255 - d_bg_color.r),
            (unsigned char) (255 - d_bg_color.g),
            (unsigned char) (255 - d_bg_color.b),
            255
    };
    if (tx < d_width && ty < d_height) {
        surf2Dwrite(data, d_surfaceObject, tx * sizeof(uchar4), ty);
    }
}

void update()
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

    update_kernel<<<gridSize, blockSize>>>(surface_object);
    getLastCudaError("update_kernel error");

    cudaDestroySurfaceObject(surface_object);
    cudaGraphicsUnmapResources(1, &resource);
    cudaDeviceSynchronize();
}
