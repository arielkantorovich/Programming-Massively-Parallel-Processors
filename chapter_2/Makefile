# ==========================================================
# chapter_2 - CUDA + host C++ build
# Builds:  main.cpp + src/*.cpp + src/*.cu   ->   bin/main
# ==========================================================

# ----- Automatic Variables ------------------
# - $@ = the target being built
# - $< = the first prerequisite (the source file)
# - $^ = all prerequisites (all the .o files)
# - $(dir $@) = directory part of the target
# --------------------------------------------

# ---- 1. Toolchain ----
# We drive the WHOLE build through nvcc (not g++) because:
#   * src/vecAdd.cu contains device code -> only nvcc understands <<<>>> and __global__
#   * main.cpp #includes vecAdd.cuh -> which pulls in <cuda_runtime.h>
#   * the final link must pull in the CUDA runtime (-lcudart), which nvcc adds for us
# nvcc compiles host (.cpp) code by forwarding it to g++ under the hood.
NVCC      := nvcc
ARCH      := sm_86                 # RTX 2050 = compute capability 8.6. Override: make ARCH=sm_75
NVCCFLAGS := -std=c++17 -O2 -arch=$(ARCH) -Iincludes
#            |          |   |              └ where to find vecAdd.hpp / vecAdd.cuh
#            |          |   └ generate SASS for this exact GPU architecture
#            |          └ optimization level
#            └ C++ standard

# ---- 2. Project layout ----
SRC_DIR  := src
INC_DIR  := includes
BIN_DIR  := bin

# ---- 3. Collect sources, derive object + exe names ----
CPP_SRCS := main.cpp $(wildcard $(SRC_DIR)/*.cpp)   # host C++ sources
CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)             # CUDA device sources
SRCS     := $(CPP_SRCS) $(CU_SRCS)
# Keep the original extension inside the object name so that
# src/vecAdd.cpp and src/vecAdd.cu do NOT both map to the same vecAdd.o:
#   main.cpp        -> bin/main.cpp.o
#   src/vecAdd.cpp  -> bin/src/vecAdd.cpp.o
#   src/vecAdd.cu   -> bin/src/vecAdd.cu.o
OBJS     := $(SRCS:%=$(BIN_DIR)/%.o)
HEADERS  := $(wildcard $(INC_DIR)/*.hpp) $(wildcard $(INC_DIR)/*.cuh)
TARGET   := $(BIN_DIR)/main                          # final program

# ---- 4. Targets that are NOT real files ----
.PHONY: all run clean
# Default goal: just running `make` builds the program
all: $(TARGET)

# ---- 5. LINK: stitch all .o files into one executable (nvcc adds -lcudart) ----
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# ---- 6. COMPILE: one source -> one .o ----
# Any header change forces a rebuild (HEADERS listed as a prerequisite).
$(BIN_DIR)/%.cpp.o: %.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/%.cu.o: %.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ---- 7. Convenience helpers ----
run: $(TARGET)
	./$(TARGET)
clean:
	rm -rf $(BIN_DIR)
