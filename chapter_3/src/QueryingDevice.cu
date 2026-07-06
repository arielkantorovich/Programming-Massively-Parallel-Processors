#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count); // case several devices
    cudaDeviceProp dev_prop; // struct that contain all information see https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp

    for (int i=0; i < dev_count; i++) // run on each device print the hardware information
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
        printf("number of registers that are available in each SM: %i \n", dev_prop.regsPerMultiprocessor);
        printf("sharedMemPerBlock: %zu [bytes] \n", dev_prop.sharedMemPerBlock);
        printf("sharedMemPerMultiprocessor: %zu [bytes]\n", dev_prop.sharedMemPerMultiprocessor);
    }

    return 0;
}