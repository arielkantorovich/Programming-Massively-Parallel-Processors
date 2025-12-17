#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "param_c.hpp"

/**
 * @brief Implement naive stencil kernel
 * @param in - input 3D size (n, n, n)
 * @param out - output 3D size (n, n ,n)
 * @param N - size in each axes
*/
__global__ void stencill_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= 1 && i < N-1 && j >=1 && j < N-1 && k>=1 && k < N-1)
    {
        out[k + j * N + i * N * N] = c0 * in[k + j * N + i * N * N] + 
                                     c1 * in[(k - 1) + j * N + i * N * N] + 
                                     c2 * in[(k + 1) + j * N + i * N * N] + 
                                     c3 * in[k + (j - 1) * N + i * N * N] +
                                     c4 * in[k + (j + 1) * N + i * N * N] +
                                     c5 * in[k + j * N + (i - 1) * N * N] + 
                                     c6 * in[k + j * N + (i + 1) * N * N];
    }
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error need run ./<exe file> <n> where <n> is the size in each 3D axes. \n");
        return -1;
    }

    int n = atoi(argv[1]);
    size_t SizeInBytes = n * n * n * sizeof(float);

    // Allocate Memory in Host painned memory (pageable locked)
    float *h_in, *h_out;
    cudaMallocHost((void**)&h_in, SizeInBytes);
    cudaMallocHost((void**)&h_out, SizeInBytes);
    putConst(h_in, 2.3f, n);

    // Allocate device Memo
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, SizeInBytes);
    cudaMalloc((void**)&d_out, SizeInBytes);

    // Transfer dara from host to device
    cudaMemcpy(d_in, h_in, SizeInBytes, cudaMemcpyHostToDevice);

    // Define Grid parameters
    dim3 BlockSize(8, 8, 8);
    dim3 GridSize;
    GridSize.x = (n + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (n + BlockSize.y - 1) / BlockSize.y;
    GridSize.z = (n + BlockSize.z - 1) / BlockSize.z;

    // Launch Kernel
    stencill_kernel <<< GridSize, BlockSize >>> (d_in, d_out, n);
    cudaDeviceSynchronize();

    // return results to host
    cudaMemcpy(h_out, d_out, SizeInBytes, cudaMemcpyDeviceToHost);
    
    // Print results
    PrintResult(h_out, n);

    // reallocate memory
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);

    return 0;
}