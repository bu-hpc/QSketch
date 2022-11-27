# CUDA= -lcurand
# CXX = nvcc

# HEADER_FILES = $(wildcard lib/*.h)
# TEST_FILES = $(wildcard *.cu)
# NAMES = $(patsubst %.cu, %, $(TEST_FILES))
CUDA_ROOT_DIR = /usr/local/cuda

CC = g++
NVCC = nvcc
CUDA_LINK_LIBS = -lcurand

SRC_DIR = src
BIN_DIR = bin
INC_DIR = include
LIB_DIR = lib

# NVCC_FLAGS = --gpu-architecture=sm_70
NVCC_FLAGS = --gpu-architecture=sm_60
# NVCC_FLAGS = -forward-unknown-to-host-compiler -Wunused
# NVCC_FLAGS = -G
NVCC_FLAGS += -rdc=true
LIB_SOURCE_CODE = $(wildcard $(LIB_DIR)/*.cu)
CUDA_SOURCE_CODE = $(wildcard $(SRC_DIR)/*.cu)
HEADER_FILES = $(wildcard $(INC_DIR)/*.h)

# EXE = run

all : $(BIN_DIR)/test $(BIN_DIR)/main $(BIN_DIR)/global_memory_perf

$(BIN_DIR)/test : $(LIB_SOURCE_CODE) $(HEADER_FILES) $(SRC_DIR)/test.cu
	$(NVCC) $(LIB_SOURCE_CODE) $(SRC_DIR)/test.cu $(NVCC_FLAGS) -o $@ -I$(INC_DIR) $(CUDA_LINK_LIBS)

$(BIN_DIR)/main : $(LIB_SOURCE_CODE) $(HEADER_FILES) $(SRC_DIR)/main.cu
	$(NVCC) $(LIB_SOURCE_CODE) $(SRC_DIR)/main.cu $(NVCC_FLAGS) -o $@ -I$(INC_DIR) $(CUDA_LINK_LIBS)

$(BIN_DIR)/global_memory_perf : $(LIB_SOURCE_CODE) $(HEADER_FILES) $(SRC_DIR)/global_memory_perf.cu
	$(NVCC) $(LIB_SOURCE_CODE) $(SRC_DIR)/global_memory_perf.cu $(NVCC_FLAGS) -o $@ -I$(INC_DIR) $(CUDA_LINK_LIBS)

clean :
	-rm $(BIN_DIR)/*