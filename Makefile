CUDA_HOME := /usr/local/cuda-10.1
CC        := g++
NVCC      := $(CUDA_HOME)/bin/nvcc
FLAGS     := -std=c++14
FLAGS_CU  := $(FLAGS) --ptxas-options=-v -arch=sm_50 --use_fast_math
LIB       := -lGLEW -lGL -lglfw -lyaml-cpp
LIB_CUDA  := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC       := -I$(CUDA_HOME)/include -I.

all: ray-tracer-cpu ray-tracer-cuda

HEADERS      := shader-program.h update.h surface.h light.h scene.h
HEADERS_CUDA := $(HEADERS) helper_cuda_opengl.h
OBJ          := ray-tracer.o shader-program.o surface.o light.o scene.o
OBJ_CPU      := $(OBJ) update-cpu.o
OBJ_CUDA     := $(OBJ) update-cuda.o

%.o: %.cpp Makefile
	$(CC) $(FLAGS) -c -o $@ $< $(INC)

%.o: %.cu Makefile
	$(NVCC) $(FLAGS_CU) -dc -o $@ $< $(INC)

ray-tracer-cpu: $(OBJ_CPU) $(HEADERS) Makefile
	$(NVCC) $(FLAGS_CU) -o $@ $(OBJ_CPU) $(LIB_CUDA)

ray-tracer-cuda: $(OBJ_CUDA) $(HEADERS_CUDA) Makefile
	$(NVCC) $(FLAGS_CU) -o $@ $(OBJ_CUDA) $(LIB_CUDA)

clean:
	rm -f *.o ray-tracer-*
