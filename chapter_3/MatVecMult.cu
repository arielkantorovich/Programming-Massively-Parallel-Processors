#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * @brief  A matrix–vector multiplication takes an input matrix B 
 * and a vector C and produces one output vector A.
 * @param A output vector
 * @param B square matrix transformation
 * @param C inpute Vector
*/
__global__
void MatVecMult(float *A, float *B, float *C, int n)
{
    int Row = threadIdx.x + blockDim.x * blockIdx.x;
    if (Row < n)
    {
        float sum = 0.0f;
        for (int Col=0; Col < n; Col++)
        {
            sum += B[Col + Row*n] * C[Col];
        }
        A[Row] = sum;
    }
}

/**
 * @brief print result vector
 * @param result - vactor
 * @param n - vector size
*/
void printResult(float *Result, int n)
{
    for (int i=0; i < n; i++)
    {
        printf("A[%i] = %f\n", i, Result[i]);
    }
}

int main(int argc, char **argv)
{
    if (argc !=2)
    {
        printf("Error need run ./exe <n> where n is the matrix and vector dim.\n");
        return 1;
    }

    int n = atoi(argv[1]);
    size_t SizeInBytesMat = n*n*sizeof(float);
    size_t SizeInBytesVec = n*sizeof(float);

    // Allocate Host Memory
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(SizeInBytesVec);
    h_B = (float *)malloc(SizeInBytesMat);
    h_C = (float *)malloc(SizeInBytesVec);

    // Put some var inside
    for (int i = 0; i < n*n; ++i) h_B[i] = 10.0f;
    for (int i = 0; i < n;   ++i) h_C[i] = 1.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, SizeInBytesVec);
    cudaMalloc((void **)&d_B, SizeInBytesMat);
    cudaMalloc((void **)&d_C, SizeInBytesVec);

    // Transfer Data from Host to Device
    cudaMemcpy(d_B, h_B, SizeInBytesMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, SizeInBytesVec, cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 BlockDim(256, 1, 1);
    dim3 GridDim;
    GridDim.x = (n + BlockDim.x - 1) / BlockDim.x;

    MatVecMult<<<GridDim, BlockDim>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    // Transfer Results back to Host
    cudaMemcpy(h_A, d_A, SizeInBytesVec, cudaMemcpyDeviceToHost);

    printResult(h_A, n);

    // release Host and device memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}