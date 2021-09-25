#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>
#include "scene-exception.h"
#include "shader-program.h"
#include "update.h"
#include "scene.h"

float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f
};

int indices[] = {
        0, 1, 3,
        0, 2, 3
};

// camera data
glm::dvec3 position(0.0);
glm::dvec3 direction(0.0);
const glm::dvec3 up(0.0, 1.0, 0.0);
glm::dvec3 movement_front(0.0);
glm::dvec3 camera_right(0.0);
glm::dvec3 camera_up(0.0);
double pitch = 0.0;
double yaw = 90.0;
const double sensitivity = 0.1;
const double camera_base_speed = 10.0;
double last_frame_time = 0.0;

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void update_direction()
{
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camera_right = -glm::normalize(glm::cross(direction, up));
    camera_up = glm::cross(direction, camera_right);
    movement_front = glm::cross(camera_right, up);
}

glm::dmat4 camera_matrix()
{
    auto world_to_camera = glm::lookAt(position, position - direction, up);
    return glm::inverse(world_to_camera);
}

void process_inputs(GLFWwindow *window)
{
    static bool capture_mouse = true;
    static bool capture_mouse_released = true;
    double delta_time = glfwGetTime() - last_frame_time;
    double camera_speed = camera_base_speed * delta_time;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        position += movement_front * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        position -= movement_front * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        position += camera_right * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        position -= camera_right * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        position.y += camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        position.y -=  camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
        if (capture_mouse_released) {
            capture_mouse_released = false;
            if (capture_mouse) {
                capture_mouse = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            else {
                capture_mouse = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
        }
    }
    else {
        capture_mouse_released = true;
    }
    last_frame_time = glfwGetTime();
}

void mouse_callback(GLFWwindow *window, double x, double y)
{
    static bool first_mouse = true;
    static double x_last, y_last;
    if (first_mouse)
    {
        x_last = x;
        y_last = y;
        first_mouse = false;
    }

    double x_offset = (x - x_last) * sensitivity;
    double y_offset = (y - y_last) * sensitivity;
    x_last = x;
    y_last = y;

    yaw   -= x_offset;
    pitch -= y_offset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;
}

int main(int argc, char *argv[])
{
    // GLFW initialization
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // load the scene
    if (argc < 2) {
        fprintf(stderr, "Input file not specified\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    Scene scene;
    try {
        scene = Scene::load_from_file(argv[1]);
    }
    catch (SceneException& e) {
        fprintf(stderr, "Error during scene loading\n%s\n", e.what());
        glfwTerminate();
        return EXIT_FAILURE;
    }

    int window_width = 800, window_height = 600;
    if (argc >= 4) {
        window_width = strtol(argv[2], nullptr, 10);
        window_height = strtol(argv[3], nullptr, 10);
        if (window_width < 10 || window_height < 10) {
            fprintf(stderr, "Invalid window size\n");
            glfwTerminate();
            return EXIT_FAILURE;
        }
    }

    // create a window
    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Ray tracer", nullptr, nullptr);
    if (window == nullptr) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // initialize GLEW and load the shader program
    glewInit();
    ShaderProgram program("ray-tracer.vert", "ray-tracer.frag");

    // initialize OpenGL objects
    unsigned int vbo;
    glGenBuffers(1, &vbo);
    unsigned int ebo;
    glGenBuffers(1, &ebo);
    unsigned int vao;
    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texture creation and initialization
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene.px_width, scene.px_height, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    init_update(texture, scene);

    last_frame_time = glfwGetTime();

    // render loop
    int frames = 0;
    auto start_time = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        update_direction();
        process_inputs(window);

        float render_time = update(camera_matrix());
        program.use();
        glBindVertexArray(vao);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
        auto elapsed = glfwGetTime() - start_time;
        if (elapsed >= 1.0) {
            printf("FPS: %.4lf, last render time: %.4f ms\n", frames / elapsed, render_time);
            frames = 0;
            start_time = glfwGetTime();
        }
    }

    cleanup_update();
    glfwTerminate();
    return EXIT_SUCCESS;
}
