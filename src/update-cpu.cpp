#include <GLFW/glfw3.h>
#include <vector>
#include <cstdio>
#include "update.h"
#include "surface_impl.h"
#include "light_impl.h"

const int NO_OBJECT = -1;

unsigned int g_texture;
int g_width, g_height;
std::vector<float> g_data;
double g_aspect_ratio, g_vertical_fov;
std::vector<Object> g_objects;
std::vector<LightSource> g_lights;
glm::vec3 g_bg_color;
const glm::dvec4 RAY_ORIGIN(0.0, 0.0, 0.0, 1.0);
glm::dvec3 g_ray_origin;
int g_max_reflections;


void init_update(unsigned int texture, const Scene &scene)
{
    g_texture = texture;
    g_width = scene.px_width;
    g_height = scene.px_height;
    g_aspect_ratio = scene.aspect_ratio();
    g_vertical_fov = tan(0.5 * scene.vertical_fov);
    g_objects = scene.objects;
    g_lights = scene.lights;
    g_bg_color = scene.bg_color;
    g_max_reflections = scene.max_reflections;

    // reserve space for the RGB data of the texture
    g_data.resize(g_width * g_height * 3);

    // fill the data buffer with the background color
    for (int i = 0; i < g_width * g_height; i++) {
        g_data[3 * i] = g_bg_color.r;
        g_data[3 * i + 1] = g_bg_color.g;
        g_data[3 * i + 2] = g_bg_color.b;
    }
}

int get_color_and_object(const glm::dvec3 &origin, const glm::dvec3 &dir, glm::vec3 &result_color,
                         glm::dvec3 &surface_point, glm::dvec3 &surface_normal)
{
    int best_idx = NO_OBJECT;
    double best_t = INFINITY;
    for (int i = 0; i < (int) g_objects.size(); i++) {
        double t = intersect_ray(g_objects[i].surface, origin, dir);
        if (t >= EPS && t < MAX_T && t < best_t) {
            best_t = t;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        result_color = glm::dvec3(0.0);
        surface_point = origin + best_t * dir;
        surface_normal = normal_vector(g_objects[best_idx].surface, surface_point);
        auto object_color = g_objects[best_idx].color;
        for (const LightSource &light: g_lights) {
            double max_t = 0;
            auto shadow_dir = shadow_ray(light, surface_point, max_t);
            bool in_shadow = false;
            for (const Object &object: g_objects) {
                double t = intersect_ray(object.surface, surface_point + SHADOW_BIAS * surface_normal, shadow_dir);
                if (t > EPS && t < max_t) {
                    in_shadow = true;
                    break;
                }
            }
            if (!in_shadow) {
                result_color += surface_color(light, surface_point, surface_normal, object_color);
            }
        }
        result_color = glm::min(glm::vec3(1.0f), result_color);
    }
    return best_idx;
}

glm::vec3 render_pixel(const glm::dmat4 &camera_matrix, int pixel_x, int pixel_y)
{
    double ndc_x = (pixel_x + 0.5) / g_width;
    double ndc_y = (pixel_y + 0.5) / g_height;
    double camera_x = (2.0 * ndc_x - 1.0) * g_aspect_ratio * g_vertical_fov;
    double camera_y = (2.0 * ndc_y - 1.0) * g_vertical_fov;
    glm::dvec3 dir(camera_x, camera_y, 1.0);
    dir = glm::normalize(glm::dvec3(camera_matrix * glm::dvec4(dir, 1.0)) - g_ray_origin);

    glm::vec3 object_color;
    glm::dvec3 surface_point, surface_normal;
    auto idx = get_color_and_object(g_ray_origin, dir, object_color, surface_point, surface_normal);
    if (idx == NO_OBJECT) {
        return g_bg_color;
    }
    auto result_color = object_color;
    float cur_ratio = 1.0f;
    int cur_reflections = 0;
#define UPDATE_COLOR(col) result_color = (1.0f - cur_ratio) * result_color + cur_ratio * (col);
    while (g_objects[idx].reflection_ratio > EPS) {
        cur_ratio *= g_objects[idx].reflection_ratio;
        if (cur_reflections == g_max_reflections) {
            UPDATE_COLOR(g_bg_color)
            break;
        }
        cur_reflections++;

        dir = reflect_ray(dir, surface_normal);
        auto origin = surface_point + SHADOW_BIAS * surface_normal;
        idx = get_color_and_object(origin, dir, object_color, surface_point, surface_normal);
        if (idx == NO_OBJECT) {
            UPDATE_COLOR(g_bg_color)
            break;
        }
        UPDATE_COLOR(object_color)
    }
    return result_color;
}

float update(const glm::dmat4 &camera_matrix)
{
    g_ray_origin = camera_matrix * RAY_ORIGIN;
    auto start_time = glfwGetTime();
    for (int y = 0; y < g_height; y++) {
        for (int x = 0; x < g_width; x++) {
            auto c = render_pixel(camera_matrix, x, y);
            size_t idx = y * g_width + x;
            g_data[3 * idx] = c.r;
            g_data[3 * idx + 1] = c.g;
            g_data[3 * idx + 2] = c.b;
        }
    }
    auto elapsed = glfwGetTime() - start_time;

    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_width, g_height, 0, GL_RGB, GL_FLOAT, &g_data[0]);
    return elapsed * 1000.0f;
}

void cleanup_update()
{

}
