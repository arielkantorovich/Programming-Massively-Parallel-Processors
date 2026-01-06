# Manual Commands

```
# Step 1: Compile CUDA kernel file
nvcc -O2 -lineinfo -arch=sm_86 -DNUM_BINS=256 -DBIN_SHIFT=0 -c histogram_cuda.cu -o histogram_cuda.o
```

```
# Step 2: Compile main file
nvcc -O2 -lineinfo -arch=sm_86 -DNUM_BINS=256 -DBIN_SHIFT=0 $(pkg-config --cflags opencv4) -c main.cu -o main.o
```
```
# Step 3: Link everything together
nvcc -O2 -lineinfo -arch=sm_86 main.o histogram_cuda.o -o histogram $(pkg-config --libs opencv4)
```