#include <GLFW/glfw3.h>
#include <vector>
#include "update.h"

unsigned int g_texture;
int g_width, g_height;
std::vector<float> g_data;

void init_update(unsigned int texture, int width, int height)
{
    g_texture = texture;
    g_width = width;
    g_height = height;

    // reserve space for the RGB data of the texture
    g_data.resize(width * height * 3);

    // for now - fill the data with white pixels
    for (float& x : g_data) {
        x = 1.0f;
    }
}

void update()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0, GL_RGB, GL_FLOAT, &g_data[0]);
}
