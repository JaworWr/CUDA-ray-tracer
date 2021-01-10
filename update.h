#ifndef CUDA_RAY_TRACER_UPDATE_H
#define CUDA_RAY_TRACER_UPDATE_H

#include "scene.h"

void init_update(unsigned int texture, const Scene& scene);
float update(const glm::dmat4& camera_matrix);

#endif //CUDA_RAY_TRACER_UPDATE_H
