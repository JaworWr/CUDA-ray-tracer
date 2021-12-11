CUDA_HOME  := /usr/local/cuda-10.2
CC         := g++
NVCC       := $(CUDA_HOME)/bin/nvcc
FLAGS      := -std=c++14
FLAGS_CUDA := $(FLAGS) --ptxas-options=-v,-warn-spills -arch=sm_50 --use_fast_math -Xcudafe="--diag_suppress=2929"
LIB        := -lGLEW -lGL -lglfw -lyaml-cpp
LIB_CUDA   := $(LIB) -L$(CUDA_HOME)/lib -lcudart
INC        := -I$(CUDA_HOME)/include -I.

BUILD_DIR      := build
BUILD_DIR_CUDA := $(BUILD_DIR)/cuda
SOURCES_CPP    := $(wildcard *.cpp)
SOURCES_CU     := $(wildcard *.cu)
OBJ_CPP        := $(SOURCES_CPP:%.cpp=$(BUILD_DIR)/%.o)
OBJ_CU         := $(SOURCES_CU:%.cu=$(BUILD_DIR_CUDA)/%.o)

EXEC_CPU   := ray-tracer-cpu
EXEC_CUDA  := ray-tracer-cuda
OBJ_CPU    := $(filter-out $(BUILD_DIR)/update-%.o,$(OBJ_CPP)) $(BUILD_DIR)/update-cpu.o
OBJ_CUDA   := $(filter-out $(BUILD_DIR)/update-%.o,$(OBJ_CPP)) $(BUILD_DIR_CUDA)/update-cuda.o

all: $(EXEC_CPU) $(EXEC_CUDA)

$(BUILD_DIR)/%.d: %.cpp Makefile
	@mkdir -p $(BUILD_DIR)
	@$(CC) -MM $(INC) $(FLAGS) $< > $@
	@sed -i 's~$(*F).o~$@ $(@:.d=.o)~' $@

$(BUILD_DIR_CUDA)/%.d: %.cu Makefile
	@mkdir -p $(BUILD_DIR_CUDA)
	@$(NVCC) -MM $(INC) $(FLAGS_CUDA) $< > $@
	@sed -i 's~$(*F).o~$@ $(@:.d=.o)~' $@

include $(OBJ_CPP:.o=.d)
include $(OBJ_CU:.o=.d)

$(BUILD_DIR)/%.o: %.cpp $(BUILD_DIR)/%.d
	$(CC) $(FLAGS) $(INC) -c -o $@ $<

$(BUILD_DIR_CUDA)/%.o: %.cu $(BUILD_DIR_CUDA)/%.d
	$(NVCC) $(FLAGS_CUDA) $(INC) -c -o $@ $(@:$(BUILD_DIR_CUDA)/%.o=%.cu)

$(EXEC_CPU): $(OBJ_CPU)
	$(CC) $(FLAGS) -o $@ $(OBJ_CPU) $(LIB)

$(EXEC_CUDA): $(OBJ_CUDA)
	$(NVCC) $(FLAGS_CUDA) -o $@ $(OBJ_CUDA) $(LIB_CUDA)

clean:
	rm -f \
		$(BUILD_DIR)/*.o \
		$(BUILD_DIR_CUDA)/*.o \
		$(EXEC_CPU) $(EXEC_CUDA)
