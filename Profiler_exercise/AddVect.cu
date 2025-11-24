#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Adding element wise vectors C = A + B
 * @param n - vector length
*/
__global__
void AddVec(float *A, float *B, float *C, unsigned int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}



int main(int argc, char **argv)
{
    unsigned int  n = 1 << 18;
    size_t SizeInBytes = n * sizeof(float);

    // Allocate host Memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(SizeInBytes);
    h_B = (float*)malloc(SizeInBytes);
    h_C = (float*)malloc(SizeInBytes);

    for (int i=0; i < n; i++)
    {
        h_A[i] = 2.5f;
        h_B[i] = 3.5f;
    }

    // Allocate device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, SizeInBytes);
    cudaMalloc((void **)&d_B, SizeInBytes);
    cudaMalloc((void **)&d_C, SizeInBytes);

    // Transfer data from 
    cudaMemcpy(d_A, h_A, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 BlockDim(256, 1, 1);
    dim3 GridDim((n - 1 + BlockDim.x) / BlockDim.x, 1, 1);

    AddVec <<< GridDim, BlockDim >>> (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Transfer Data from device to Host
    cudaMemcpy(h_C, d_C, SizeInBytes, cudaMemcpyDeviceToHost);

    // Deallocate memory
    free(h_A);  free(h_B);  free(h_C);
    cudaFree(d_A);  cudaFree(d_B); cudaFree(d_C);

    return 0;
}