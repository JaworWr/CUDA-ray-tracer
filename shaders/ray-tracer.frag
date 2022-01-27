#version 330 core

in vec2 texCoords;

uniform sampler2D texSampler;

void main() {
    gl_FragColor = texture(texSampler, texCoords);
}
