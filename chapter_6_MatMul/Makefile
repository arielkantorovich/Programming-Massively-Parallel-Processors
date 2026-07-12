# ========= Compilers & Flags =========
NVCC      := nvcc
NVCCFLAGS := -O2 -lineinfo -arch=sm_86 -Iincludes

# ========= Sources / target =========
SRC     := src/GEMM.cu src/main.cu
TARGET  := bin/gemm

.PHONY: all clean run
all: $(TARGET)

$(TARGET): $(SRC) includes/GEMM.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
