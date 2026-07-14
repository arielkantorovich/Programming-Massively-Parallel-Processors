# ========= Compilers & Flags =========
NVCC      := nvcc
NVCCFLAGS := -O2 -lineinfo -arch=sm_86 -Xptxas -v -Iincludes

# ========= Sources / target =========
SRC       := src/GEMM.cu src/main.cu
TARGET    := bin/gemm

QUARY_SRC    := src/QuaryDevice.cu
QUARY_TARGET := bin/quaryDev

.PHONY: all clean run run-quary
all: $(TARGET) $(QUARY_TARGET)

$(TARGET): $(SRC) includes/GEMM.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

$(QUARY_TARGET): $(QUARY_SRC)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(QUARY_SRC) -o $(QUARY_TARGET)

run: $(TARGET)
	./$(TARGET)

run-quary: $(QUARY_TARGET)
	./$(QUARY_TARGET)

clean:
	rm -f $(TARGET) $(QUARY_TARGET)
