#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Put constant inside all elements in matrix
 * @param M - matrix Size (n, n) 
 * @param n - size
 * @param con - constant
*/
__host__
void SetConstant(float *M, int n, float con)
{
    for (int i = 0; i < n * n; i++)
    {
        M[i] = con;
    }
}


/**
 * @brief print matrix elements on host
 * @param M - input matrix
 * @param w - col/row of matrix
 * @note we solve square matrix.
*/
__host__
void PrintResults(float *M, int w)
{
    for (int i=0; i < w; i++)
    {
        for (int j=0; j < w; j++)
        {
            printf("M[%i][%i] = %f", i, j, M[j + i * w]);
        }
    }
}

/**
 * @brief Naive matrix multiplication C=AB
 * @param A mxm input matrix
 * @param B mxm input matrix
 * @param C mxm output matrix
*/
__global__
void MatMul(float *A, float *B, float *C, int w)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Row >= w || Col >= w)
        return;
    
    float sum = 0.0f;
    for (int k = 0; k < w; k++)
    {
        sum += A[k + Row * w] * B[Col + w * k];
    }

    C[Col + Row*w] = sum;
}


int main(int argc, char **argv)
{
    if (argc !=2)
    {
        printf("Error: argc!=2, need run ./<exe> <n> where n is the matrix size. \n");
        return -1;
    }

    int n = atoi(argv[1]);
    size_t SizeInBytes = n * n * sizeof(float);

    // Define Memory in Host
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(SizeInBytes);
    h_B = (float *)malloc(SizeInBytes);
    h_C = (float *)malloc(SizeInBytes);

    // Put constant in matrices inpput
    SetConstant(h_A, n, 0.25f);
    SetConstant(h_B, n, 2.0f);

    // Define Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, SizeInBytes);
    cudaMalloc((void **) &d_B, SizeInBytes);
    cudaMalloc((void **) &d_C, SizeInBytes);

    // Define Grid Kernel Parameters
    unsigned int ThredinBlock = 16;
    dim3 GridSize;
    dim3 BlockSize(ThredinBlock, ThredinBlock, 1);
    GridSize.x = (n + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (n + BlockSize.y - 1) / BlockSize.y;
    GridSize.z = 1;

    // Transfer Data from Host to Device
    cudaMemcpy(d_A, h_A, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch Kernel
    MatMul <<< GridSize, BlockSize >>> (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Move Memory from host to device
    cudaMemcpy(h_C, d_C, SizeInBytes, cudaMemcpyDeviceToHost);

    // Release memory
    free(h_A);  free(h_B);  free(h_C);
    cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);

    return 0;
}