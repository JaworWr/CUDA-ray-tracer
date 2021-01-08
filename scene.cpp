#include <cmath>
#include "scene.h"

Scene::Scene(int px_width, int px_height, double vertical_fov_deg, const glm::vec3 &bg_color)
:px_width{px_width}, px_height{px_height}, bg_color{bg_color}, objects{}
{
    vertical_fov = vertical_fov_deg * M_PI / 180.0;
}
