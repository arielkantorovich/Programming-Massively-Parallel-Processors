# Preformense consideration
To check and anaylze our algorithms such shared memory consideration kernel size etc. We going to use NVIDIA profiler tools such:
1. NVIDIA Nsight Systems (Dive Analyze to kernel)
2. NVIDIA Nsight Compute (Give memory observation and time dealys)

### Example
1. Compile .cu file using nvcc compiler
```
nvcc -O2 -arch=sm_xy -lineinfo name.cu -o name
```
2. Generate nsight compute report
```
ncu -o report <cuda exe>
```
3. Generate nsight system report
```
nsys profile -o report <cuda exe>
```

* Now we can analyze the reports using nsys and ncu on windows computer or in linux use gui such
```
ncu-ui
```

```
nsys-ui
```