#include "scene-exception.h"

void validate_color(const glm::vec3 &color)
{
#define BAD_COORD(c) (color.c < 0.0f || color.c > 1.0f)
    if (BAD_COORD(x) || BAD_COORD(y) || BAD_COORD(z)) {
        std::stringstream error;
        error << "Invalid color: (" << color.x << ", " << color.y << ", " << color.z << ")";
        throw SceneException(error.str());
    }
}