CUDA_HOME := /usr/local/cuda-10.1
CC        := g++
NVCC      := $(CUDA_HOME)/bin/nvcc
LIB       := -lGLEW -lGL -lglfw
LIB_CUDA  := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC       :=
INC_CUDA  := $(INC) -I$(CUDA_HOME)/include -I.

all: ray-tracer-cpu ray-tracer-cuda

HEADERS      := shader-program.h update.h
HEADERS_CUDA := $(HEADERS) helper_cuda_opengl.h
OBJ          := ray-tracer.o shader-program.o
OBJ_CPU      := $(OBJ) update-cpu.o
OBJ_CUDA     := $(OBJ) update-cuda.o

%.o: %.cpp
	$(CC) -c -o $@ $< $(INC)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(INC_CUDA)

ray-tracer-cpu: $(OBJ_CPU) $(HEADERS)
	$(CC) -o $@ $(OBJ_CPU) $(LIB)

ray-tracer-cuda: $(OBJ_CUDA) $(HEADERS_CUDA)
	$(NVCC) -o $@ $(OBJ_CUDA) $(LIB_CUDA)

clean:
	rm -f *.o ray-tracer-*
