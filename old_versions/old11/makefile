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

# NVCC_FLAGS = --gpu-architecture=sm_60
# NVCC_FLAGS = -forward-unknown-to-host-compiler -Wunused
# NVCC_FLAGS = -G
CUDA_SOURCE_CODE = $(wildcard $(SRC_DIR)/*.cu)
HEADER_FILES = $(wildcard $(INC_DIR)/*.h)

EXE = run


$(BIN_DIR)/$(EXE) : $(CUDA_SOURCE_CODE) $(HEADER_FILES)
	$(NVCC) $(CUDA_SOURCE_CODE)  $(NVCC_FLAGS) -o $@ -I$(INC_DIR) $(CUDA_LINK_LIBS)

clean :
	-rm $(BIN_DIR)/*