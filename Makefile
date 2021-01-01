CUDA_HOME := /usr/local/cuda-10.1
CC    := g++
NVCC  := $(CUDA_HOME)/bin/nvcc
LIB   := -lGLEW -lGL -lglfw
LIBCU := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC   :=
INCCU := $(INC) -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc

all: ray-tracer-cpu

HEADERS :=
OBJ := ray-tracer.o

%.o: %.cpp
	$(CC) -c -o $@ $< $(INC)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(INCCU)

ray-tracer-cpu: $(OBJ) $(HEADERS)
	$(NVCC) -o $@ $(OBJ) $(LIBCU)

clean:
	rm -f $(OBJ) ray-tracer-cpu
