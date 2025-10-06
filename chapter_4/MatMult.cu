#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/**
 * @brief Matrix Multiplication P = MN
 * @param M - input matrix size (I, J)
 * @param N - input matrix size (J, L)
 * @param P - output matrix result (I, L)
 * @note for simplicy, implementation assume sqaure matrices
*/
__global__
void MatrixMultiplication(float *M, float *N, float *P, int w)
{
    int Row = threadIdx.y + blockDim.y * blockIdx.y;
    int Col = threadIdx.x + blockDim.x * blockIdx.x;

    if (Row < w && Col < w)
    {
        float Pval = 0;
        for (int k=0; k < w; k++)
        {
            Pval += M[Row*w + k] * N[w*k + Col]; 
        }
        P[Col + Row*w] = Pval;
    }

}

void PrintResult(float *P, int n)
{
    for (int i=0; i < n; i++)
    {
        for (int j=0; j < n; j++)
        {
            printf("P[%i][%i] = %f \n", i, j, P[j+i*n]);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc!=2)
    {
        printf("Error: Run ./exe <n> where n is matrices size.\n");
        return 1;
    }

    int n = atoi(argv[1]);

    // Define Host memo
    float *h_P, *h_N, *h_M;
    size_t SizeInBytes = n *n * sizeof(float);

    h_M = (float*)malloc(SizeInBytes);
    h_N = (float*)malloc(SizeInBytes);
    h_P = (float*)malloc(SizeInBytes);

    for (int i=0 ; i < n*n; i++)
    {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    // Allocate Gpu Memory
    float *d_P, *d_M, *d_N;
    cudaMalloc((void **)&d_M, SizeInBytes);
    cudaMalloc((void **)&d_N, SizeInBytes);
    cudaMalloc((void **)&d_P, SizeInBytes);
    
    // Transfer Data from Host to Device
    cudaMemcpy(d_M, h_M, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, SizeInBytes, cudaMemcpyHostToDevice);
    
    // Launch Kernel
    dim3 BlockDim(16, 16);
    dim3 GridDim;
    GridDim.x = (n + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (n + BlockDim.y - 1) / BlockDim.y;
    MatrixMultiplication <<<GridDim, BlockDim>>>(d_M, d_N, d_P, n);

    // Transfer resulst from Host to Device
    cudaMemcpy(h_P, d_P, SizeInBytes, cudaMemcpyDeviceToHost);

    // Check reults
    PrintResult(h_P, n);

    // Release Memory Host and Device
    free(h_M); free(h_N); free(h_P);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);

    return 0;
}