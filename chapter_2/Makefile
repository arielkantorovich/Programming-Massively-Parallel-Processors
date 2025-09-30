# ==========================
# Compiler and flags
# ==========================
CC      := gcc
NVCC    := nvcc

CFLAGS  := -std=c11 -O2 -Wall -Wextra
NVFLAGS := -O3     # you can add --compiler-options '-Wall' if you want

# Sources and targets
CPU_SRC := vecAdd.c
CPU_EXE := addVec_CPU

CUDA_SRC := vecAdd.cu
CUDA_EXE := addVec_CUDA

.PHONY: all clean run_cpu run_cuda

# ==========================
# Default target
# ==========================
all: $(CPU_EXE) $(CUDA_EXE)

# ==========================
# CPU build
# ==========================
$(CPU_EXE): $(CPU_SRC)
	$(CC) $(CFLAGS) $< -o $@

# ==========================
# CUDA build
# ==========================
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(NVFLAGS) $< -o $@

# ==========================
# Run helpers
# ==========================
run_cpu: $(CPU_EXE)
	./$(CPU_EXE)

run_cuda: $(CUDA_EXE)
	./$(CUDA_EXE)

# ==========================
# Cleanup
# ==========================
clean:
	rm -f $(CPU_EXE) $(CUDA_EXE) *.o
