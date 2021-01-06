#include <GLFW/glfw3.h>
#include <vector>
#include "update.h"

unsigned int g_texture;
int g_width, g_height;
std::vector<float> g_data;
float g_aspect_ratio, g_vertical_fov;
std::vector<Object> g_objects;
glm::vec3 g_bg_color;


void init_update(unsigned int texture, const Scene& scene)
{
    g_texture = texture;
    g_width = scene.px_width;
    g_height = scene.px_height;
    g_aspect_ratio = scene.aspect_ratio();
    g_vertical_fov = scene.vertical_fov;
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

void update()
{
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0, GL_RGB, GL_FLOAT, &g_data[0]);
}
