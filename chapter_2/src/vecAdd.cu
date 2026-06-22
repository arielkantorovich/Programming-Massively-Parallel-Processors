#include "vecAdd.cuh"
#include <stdio.h>


/**
 * @brief Compare host and device results print true if vector 
 * equals and false other.
 * @param h_a input vector host results
 * @param d_a input vector device results
 */
void CompareHostDeviceResults(float *h_a, float *d_a, unsigned int n)
{
    for (unsigned int i=0; i < n; i++)
    {
        if (h_a[i] != d_a[i])
        {
            printf("Error: host and device results do not match at index %d! host: %f, device: %f\n", i, h_a[i], d_a[i]);
            return ;
        }
    }
    printf("func CompareHostDeviceResults: Succeed!\n");
    return ;
}

/**
 * @brief Host addVecGpu part, allocate ana manage nenort and transfer data. calling addVec kernel and return results to h_c.
 * @param h_a Input vector size n.
 * @param h_b Input vector size n.
 * @param h_c Output vector size n.
 * @param n vector length.
 */
void addVecDevice(float *h_a, float *h_b, float *h_c, unsigned int n)
{
    // Define device grid parameters
    unsigned int ThreadPerBlock = 256;
    unsigned int NumOfBlocks = (n - 1 + ThreadPerBlock) / ThreadPerBlock;

    // allocate device Memory
    float *d_a, *d_b, *d_c;
    size_t SizeInBytes = sizeof(float) * n;
    cudaMalloc((void**)&d_a, SizeInBytes);
    cudaMalloc((void**)&d_b, SizeInBytes);
    cudaMalloc((void**)&d_c, SizeInBytes);

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch Kernel and wait until device finish
    addVecKernel<<<NumOfBlocks, ThreadPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Transfer results from device to host
    cudaMemcpy(h_c, d_c, SizeInBytes, cudaMemcpyDeviceToHost);

    // Deaalocate device Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__
void addVecKernel(float *A, float *B, float *C, unsigned int n)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}