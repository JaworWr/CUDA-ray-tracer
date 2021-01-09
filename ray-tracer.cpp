#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include "shader-program.h"
#include "update.h"
#include "scene.h"

float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f
};

int indices[] = {
        0, 1, 3,
        0, 2, 3
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void process_inputs(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main(int argc, char *argv[])
{
    // GLFW initialization
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create a window
    GLFWwindow *window = glfwCreateWindow(800, 600, "Ray tracer", nullptr, nullptr);
    if (window == nullptr)
    {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

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

    // example scene
    Scene scene(800, 600, 45.0f, glm::vec3(0.0f, 0.1f, 0.2f));
    scene.objects.push_back({
                                    SurfaceCoefs::sphere(glm::dvec3(0, 0, 5), 1),
                                    glm::vec3(1.0f, 0.0f, 0.0f)
    });
    scene.objects.push_back({
                                    SurfaceCoefs::sphere(glm::dvec3(1, 1, 10), 1.5),
                                    glm::vec3(1.0f, 1.0f, 0.0f)
    });

    // texture creation and initialization
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene.px_width, scene.px_height, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    init_update(texture, scene);

    // render loop
    int frames = 0;
    auto start_time = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        process_inputs(window);
        update();
        program.use();
        glBindVertexArray(vao);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frames++;
        auto elapsed = glfwGetTime() - start_time;
        if (elapsed >= 1.0) {
            printf("FPS: %lf\n", frames / elapsed);
            frames = 0;
            start_time = glfwGetTime();
        }
    }

    glfwTerminate();
    return EXIT_SUCCESS;
}
