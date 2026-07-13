#include <cstdio>
#include <cuda_runtime.h>

int main(){
  int dev=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
  printf("Device            : %s\n", p.name);
  printf("Compute cap       : %d.%d\n", p.major, p.minor);
  printf("SMs               : %d\n", p.multiProcessorCount);
  printf("Warp size         : %d\n", p.warpSize);
  printf("Max threads/block : %d\n", p.maxThreadsPerBlock);
  printf("Max threads/SM    : %d\n", p.maxThreadsPerMultiProcessor);
  printf("Max blocks/SM     : %d\n", p.maxBlocksPerMultiProcessor);
  printf("Regs/block        : %d\n", p.regsPerBlock);
  printf("Regs/SM           : %d\n", p.regsPerMultiprocessor);
  printf("Shared/block(opt) : %zu B\n", p.sharedMemPerBlockOptin);
  printf("Shared/block      : %zu B\n", p.sharedMemPerBlock);
  printf("Shared/SM         : %zu B\n", p.sharedMemPerMultiprocessor);
  printf("L2 cache          : %d B\n", p.l2CacheSize);
  printf("Global mem        : %.1f GB\n", p.totalGlobalMem/1e9);
  printf("Mem clock         : %d kHz\n", p.memoryClockRate);
  printf("Mem bus width     : %d bit\n", p.memoryBusWidth);
  printf("Peak BW (approx)  : %.1f GB/s\n", 2.0*p.memoryClockRate*1e3*(p.memoryBusWidth/8)/1e9);
  return 0;
}