CUDA_HOME := /usr/local/cuda-10.2
CC        := g++
NVCC      := $(CUDA_HOME)/bin/nvcc
FLAGS     := -std=c++14
FLAGS_CU  := $(FLAGS) --ptxas-options=-v,-warn-spills -arch=sm_50 --use_fast_math -Xcudafe="--diag_suppress=2929"
LIB       := -lGLEW -lGL -lglfw -lyaml-cpp
LIB_CUDA  := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC       := -I$(CUDA_HOME)/include -I.

all: ray-tracer-cpu ray-tracer-cuda

HEADERS_CUDA := $(HEADERS) helper_cuda_opengl.h
OBJ          := ray-tracer.o shader-program.o surface.o light.o scene.o
OBJ_CPU      := $(OBJ) update-cpu.o
OBJ_CUDA     := $(OBJ) update-cuda.o

%.o: %.cpp %.h Makefile
	$(CC) $(FLAGS) -c -o $@ $< $(INC)

UPDATE_HEADERS := update.h surface_impl.h light_impl.h

update-cpu.o: update-cpu.cpp $(UPDATE_HEADERS) Makefile
	$(CC) $(FLAGS) -c -o $@ $< $(INC)

update-cuda.o: update-cuda.cu $(UPDATE_HEADERS) Makefile
	$(NVCC) $(FLAGS_CU) -dc -o $@ $< $(INC)

ray-tracer.o: ray-tracer.cpp Makefile
	$(CC) $(FLAGS) -c -o $@ $< $(INC)

ray-tracer-cpu: $(OBJ_CPU) $(HEADERS) Makefile
	$(NVCC) $(FLAGS_CU) -o $@ $(OBJ_CPU) $(LIB_CUDA)

ray-tracer-cuda: $(OBJ_CUDA) $(HEADERS_CUDA) Makefile
	$(NVCC) $(FLAGS_CU) -o $@ $(OBJ_CUDA) $(LIB_CUDA)

clean:
	rm -f *.o ray-tracer-*
