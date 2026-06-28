# ========= Compilers & Flags =========
NVCC      := nvcc
CXX       := g++
NVCCFLAGS := -O2 -arch=sm_86
CXXFLAGS  := -O2
INCLUDES  := -Iincludes

# OpenCV: use pkg-config when available (host); otherwise fall back to the
# container's source-built install in /usr/local (which has no opencv4.pc).
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/local/include/opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || echo -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui)

# ========= Binaries =========
BIN := bin

# 1) Image-processing pipeline (CUDA + OpenCV) -- UNCHANGED
APP          := $(BIN)/imgproc
APP_SOURCES  := main.cpp src/ImageProceesing.cu
APP_HEADERS  := includes/ImageProcessing.cuh

# 2) CPU box-filter blur (OpenCV C++ API, no CUDA)
BLUR         := $(BIN)/blur_cpu
BLUR_SOURCES := src/ImgBlur_cpu.c

# 3) Device query (CUDA only, no OpenCV)
QUERY         := $(BIN)/device_query
QUERY_SOURCES := src/QueryingDevice.cu

.PHONY: all clean
all: $(APP) $(BLUR) $(QUERY)

# nvcc compiles the .cu (device) and forwards main.cpp to host g++, then links.
$(APP): $(APP_SOURCES) $(APP_HEADERS) | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OPENCV_CFLAGS) $(APP_SOURCES) -o $@ $(OPENCV_LIBS)

# .c file actually uses the OpenCV C++ API -> force C++ with -x c++.
$(BLUR): $(BLUR_SOURCES) | $(BIN)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -x c++ $(BLUR_SOURCES) -o $@ $(OPENCV_LIBS)

# Pure CUDA, no OpenCV.
$(QUERY): $(QUERY_SOURCES) | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(QUERY_SOURCES) -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f $(APP) $(BLUR) $(QUERY)
