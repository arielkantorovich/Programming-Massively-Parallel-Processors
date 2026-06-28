#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaDeviceProp dev_prop;

    for (int i=0; i < dev_count; i++)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Device: %s \n", dev_prop.name);
        printf("maxThreadsPerBlock: %i \n", dev_prop.maxThreadsPerBlock);
        printf("maxThreadsPerMultiProcessor: %i \n", dev_prop.maxThreadsPerMultiProcessor);
        printf("number of SMs: %i \n", dev_prop.multiProcessorCount);
        printf("WarpSize: %i \n", dev_prop.warpSize);
        printf("maxBlocksPerMultiProcessor: %i \n", dev_prop.maxBlocksPerMultiProcessor);
        printf("maxThread in dim: (%i, %i, %i) \n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
        printf("maxBlock in dim: (%i, %i, %i) \n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    }

    return 0;
}