#include <cuda.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <cstdio>
#include "helper_cuda_opengl.h"

cudaGraphicsResource_t resource;
size_t h_width, h_height;
__constant__ size_t d_width, d_height;

size_t idiv(size_t a, size_t b) {
    return a % b == 0 ? a / b : a / b + 1;
}

void init_update(unsigned int texture, int width, int height)
{
    checkCudaErrors( cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone) );
    h_width = width;
    h_height = height;
    checkCudaErrors( cudaMemcpyToSymbol(d_width, &h_width, sizeof(int), 0, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpyToSymbol(d_height, &h_height, sizeof(int), 0, cudaMemcpyHostToDevice) );
}

__global__ void update_kernel(cudaSurfaceObject_t d_surfaceObject)
{
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    // dummy update - set teh pixel to green
    uchar4 data = {0, 255, 0, 255};
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

    cudaDestroySurfaceObject(surface_object);
    cudaGraphicsUnmapResources(1, &resource);
    cudaDeviceSynchronize();
}
