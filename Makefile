CUDA_HOME := /usr/local/cuda-10.1
CC    := g++
NVCC  := $(CUDA_HOME)/bin/nvcc
LIB   := -lGLEW -lGL -lglfw
LIBCU := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC   :=
INCCU := $(INC) -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc

all: ray-tracer-cpu

HEADERS := shader-program.h update.h
OBJ_COMMON := ray-tracer.o shader-program.o
OBJ_CPU := $(OBJ_COMMON) update-cpu.o

%.o: %.cpp
	$(CC) -c -o $@ $< $(INC)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(INCCU)

ray-tracer-cpu: $(OBJ_CPU) $(HEADERS)
	$(NVCC) -o $@ $(OBJ_CPU) $(LIBCU)

clean:
	rm -f $(OBJ) ray-tracer-cpu
