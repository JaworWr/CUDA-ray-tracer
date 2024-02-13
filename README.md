# CUDA-ray-tracer
A simple real-time ray tracer written using OpenGL and CUDA

## Compilation
Make sure you have CMake and CUDA installed. Then run:
```bash
mkdir build
cd build
cmake ..
make
```

## Running
```bash
ray-tracer <scene-file> [window-width window-height]
```
Scenes are defined in YAML, examples can be found in the directory `scenes`.

## Controls
* Look around with mouse
* Use WSAD controls for horizontal movements, and Q and Z to move up and down respectively
* Press M to disable / enable mouse cursor
* Press Escape to quit the application
