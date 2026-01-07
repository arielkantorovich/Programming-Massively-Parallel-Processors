# ========= Compilers & Flags =========
NVCC      := nvcc
NVCCFLAGS := -O2 -lineinfo -arch=sm_86

# ========= Compile-time configuration =========
# You can override these on the command line, e.g.:
#   make NUM_BINS=128 BIN_SHIFT=1
NUM_BINS  ?= 256
BIN_SHIFT ?= 0

# ========= OpenCV configuration =========
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# ========= Source files =========
TARGET      := histogram
CUDA_SRCS   := main.cu histogram_cuda.cu
OBJECTS     := main.o histogram_cuda.o
HEADERS     := histogram_cuda.cuh

# ========= Combined flags =========
COMPILE_FLAGS := $(NVCCFLAGS) \
                 -DNUM_BINS=$(NUM_BINS) \
                 -DBIN_SHIFT=$(BIN_SHIFT) \
                 $(OPENCV_CFLAGS)

LINK_FLAGS := $(OPENCV_LIBS)

# ========= Targets =========
.PHONY: all clean run

all: $(TARGET)

# Step 1: Compile histogram_cuda.cu to object file
histogram_cuda.o: histogram_cuda.cu $(HEADERS)
	$(NVCC) $(COMPILE_FLAGS) -c histogram_cuda.cu -o histogram_cuda.o

# Step 2: Compile main.cu to object file
main.o: main.cu $(HEADERS)
	$(NVCC) $(COMPILE_FLAGS) -c main.cu -o main.o

# Step 3: Link all object files together
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $(TARGET) $(LINK_FLAGS)

clean:
	rm -f $(TARGET) $(OBJECTS)

# Helper target to run with an example image
# Usage: make run IMG=path/to/image.jpg
run: $(TARGET)
	@if [ -z "$(IMG)" ]; then \
		echo "Error: Please specify an image path with IMG=path/to/image.jpg"; \
		exit 1; \
	fi
	./$(TARGET) $(IMG)

# Display current configuration
info:
	@echo "Configuration:"
	@echo "  NUM_BINS  = $(NUM_BINS)"
	@echo "  BIN_SHIFT = $(BIN_SHIFT)"
	@echo "  ARCH      = $(patsubst -arch=%,%,$(filter -arch=%,$(NVCCFLAGS)))"
	@echo "  OPENCV    = $(OPENCV_CFLAGS)"
