#include <GLFW/glfw3.h>
#include <vector>
#include <cstdio>
#include "update.h"

unsigned int g_texture;
int g_width, g_height;
std::vector<float> g_data;
double g_aspect_ratio, g_vertical_fov;
std::vector<Object> g_objects;
glm::vec3 g_bg_color;
const glm::dvec3 RAY_ORIGIN(0.0f);


void init_update(unsigned int texture, const Scene& scene)
{
    g_texture = texture;
    g_width = scene.px_width;
    g_height = scene.px_height;
    g_aspect_ratio = scene.aspect_ratio();
    g_vertical_fov = tan(0.5 * scene.vertical_fov);
    g_objects = scene.objects;
    g_bg_color = scene.bg_color;

    // reserve space for the RGB data of the texture
    g_data.resize(g_width * g_height * 3);

    // fill the data buffer with the background color
    for (int i = 0; i < g_width * g_height; i++) {
        g_data[3 * i] = g_bg_color.r;
        g_data[3 * i + 1] = g_bg_color.g;
        g_data[3 * i + 2] = g_bg_color.b;
    }
}

const glm::vec3& render_pixel(int pixel_x, int pixel_y)
{
    double ndc_x = (pixel_x + 0.5) / g_width;
    double ndc_y = (pixel_y + 0.5) / g_height;
    double camera_x = (2.0 * ndc_x - 1.0) * g_aspect_ratio * g_vertical_fov;
    double camera_y = (1.0 - 2.0 * ndc_y) * g_vertical_fov;
    glm::dvec3 dir(camera_x, camera_y, 1.0);
    dir = glm::normalize(dir);

    int best_idx = -1;
    double best_t = INFINITY;
    for (int i = 0; i < g_objects.size(); i++) {
        double t = intersect_ray(g_objects[i].surface, RAY_ORIGIN, dir);
        if (t >= EPS && t < 1e6 && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        return g_objects[best_idx].color;
    }
    else {
        return g_bg_color;
    }
}

void update()
{
    for (int y = 0; y < g_height; y++) {
        for (int x = 0; x < g_width; x++) {
            auto c = render_pixel(x, y);
            size_t idx = y * g_width + x;
            g_data[3 * idx] = c.r;
            g_data[3 * idx + 1] = c.g;
            g_data[3 * idx + 2] = c.b;
        }
    }

    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0, GL_RGB, GL_FLOAT, &g_data[0]);
}
