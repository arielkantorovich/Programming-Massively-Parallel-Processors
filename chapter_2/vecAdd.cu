#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__
void addVec(float *A, float *B, float *C, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void CheckResults(float *C, int n)
{
    for (int i=0; i <n; i++)
    {
        printf("C[%i] = %f \n", i, C[i]);
    }
}

int main(void)
{
    // Define Global Variables
    int arrSize = 1000;
    size_t sizeInBytes = arrSize * sizeof(float);
    int numOfThreads = 256; // multiply of 32, hardware efficient reasson
    int numOfBlocks = (arrSize + numOfThreads - 1) / numOfThreads;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // allocate Host Memory
    h_a = (float*) malloc(sizeInBytes);
    h_b = (float*) malloc(sizeInBytes);
    h_c = (float*) malloc(sizeInBytes);

    // Put constant inside the arrays
    for (int i=0; i < arrSize; i++)
    {
        h_a[i] = 7.7f;
        h_b[i] = 68.3f;   
    }

    // allocate Device Memory
    cudaMalloc((void **) &d_a, sizeInBytes);
    cudaMalloc((void **) &d_b, sizeInBytes);
    cudaMalloc((void **) &d_c, sizeInBytes);

    // Transfer Data from Host to Device
    cudaMemcpy(d_a, h_a, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeInBytes, cudaMemcpyHostToDevice);

    // Launch Cuda Kernel
    addVec <<<numOfBlocks, numOfThreads>>> (d_a, d_b, d_c, arrSize);

    // Wait until device finch calculation
    cudaDeviceSynchronize();

    // Transfer results from Device to Host
    cudaMemcpy(h_c, d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    CheckResults(h_c, arrSize);

    // release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // release device memory
    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c); 

    return 0;
}