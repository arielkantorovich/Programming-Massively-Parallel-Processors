# Preformense consideration
To check and anaylze our algorithms such shared memory consideration kernel size etc. We going to use NVIDIA profiler tools such:
1. NVIDIA Nsight Systems (Give memory observation and time dealys)
2. NVIDIA Nsight Compute (Dive Analyze to kernel)

### Example
1. Compile .cu file using nvcc compiler
```
nvcc -O2 -arch=sm_xy -lineinfo  -Xptxas -v name.cu -o name
```
* -O3 / -O2: Host-side optimization. Keep -O3 for host code.
* -arch=sm_xy: Generate native SASS for your GPU. Wrong arch causes JIT recompilation (slow first launch).
* -Xptxas -v: Prints registers/thread and shared-mem/block at compile time. Very useful.

2. Generate nsight system report
```
nsys profile \
  --stats=true \
  --force-overwrite=true \
  -o <report name> \
  --trace=cuda,nvtx,osrt \
  <cuda exe>
```
*  nsys profile: run the app under the system profiler.
*  --stats=true: print summary tables to the terminal after the run.
*  --force-overwrite=true: overwrite the previous report file.
*  -o <report name>: output basename.
*  --trace=cuda,nvtx,osrt: trace CUDA API + kernels, your NVTX ranges, and OS runtime (thread blocking).

3. Generate nsight compute report
```
ncu \
  --set full \
  --kernel-name <kernel/function name> \
  --launch-skip 10 \
  --launch-count 1 \
  -f -o <report name> \
  <cuda exe>
```

*  --set full: collect the full metric set (roofline, memory chart, occupancy, stalls).
*  --kernel-name: profile only this kernel (regex match).
*  --launch-skip 10: skip the first 10 launches (your warm-up).
*  --launch-count 1: profile exactly one representative invocation.
*  -f: overwrite existing report.
*  -o tile_report: write tile_report.ncu-rep (GUI).

sample version:
```
ncu -o report <cuda exe>
```

4. Now we can analyze the reports using nsys and ncu on windows computer or in linux use gui such

```
nsys-ui
```

```
ncu-ui
```