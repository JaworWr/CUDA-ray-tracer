#ifndef CUDA_RAY_TRACER_HELPER_CUDA_OPENGL_H
#define CUDA_RAY_TRACER_HELPER_CUDA_OPENGL_H

#include <GLFW/glfw3.h>
#include <helper_cuda.h>

#undef checkCudaErrors
#undef getLastCudaError

/**
 * Like check from helper_cuda.h but terminates glfw properly
 */
template <typename T>
void check_terminate(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check_terminate((val), #val, __FILE__, __LINE__)

/**
 * Like __getLastCudaError from helper_cuda.h but terminates glfw properly
 */
inline void getLastCudaError_terminate(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
}

#define getLastCudaError(msg) getLastCudaError_terminate(msg, __FILE__, __LINE__)

#endif //CUDA_RAY_TRACER_HELPER_CUDA_OPENGL_H
