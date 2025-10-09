#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define TILE_SIZE 16

/**
 * @brief Auxialry method print matrix elements
*/
void PrintResults(float *P, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("P[%i][%i]=%f \n", i, j, P[j + i * n]);
        }
    }
}


/**
 * @brief Matrix Mulatiplication using tile technique P=MN
 * @param M - input matrix size (I, I)
 * @param N - input matrix size (I, I)
 * @param P - output matrix result (I, I)
 * @note for simplicy, implementation assume sqaure matrices
*/
__global__
void TileMatMul(float *M, float *N, float *P, int n)
{
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;           int ty = threadIdx.y;
    int bx = blockIdx.x;            int by = blockIdx.y;
    int Row = ty + by*TILE_SIZE;    int Col = tx + bx*TILE_SIZE;

    float Pval = 0.0f;
    for (int ph = 0; ph < n / TILE_SIZE; ph++)
    {
        Mds[ty][tx] = M[Row*n + ph*TILE_SIZE + tx];
        Nds[ty][tx] = N[Col + n*(ph*TILE_SIZE + ty)];
        __syncthreads();

        for (int k=0; k < TILE_SIZE; k++)
        {
            Pval += Mds[ty][k]*Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row*n + Col] = Pval;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error: run ./exe <n> where n is matrix size \n");
        return -1;
    }

    int n = atoi(argv[1]);
    size_t SizeInBytes = n * n * sizeof(float);

    // Allocate Host Memo
    float *h_P, *h_N, *h_M;
    h_P = (float *)malloc(SizeInBytes);
    h_N = (float *)malloc(SizeInBytes);
    h_M = (float *)malloc(SizeInBytes);

    for (int i=0; i < n * n; i++)
    {
        h_N[i] = 1.0f;
        h_M[i] = 5.5f;
    }

    // Allocate Device Memo
    float *d_P, *d_M, *d_N;
    cudaMalloc((void **) &d_M, SizeInBytes);
    cudaMalloc((void **) &d_N, SizeInBytes);
    cudaMalloc((void **) &d_P, SizeInBytes);

    // Transfer Data from CPU to GPU
    cudaMemcpy(d_M, h_M, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 BlockDim(TILE_SIZE, TILE_SIZE);
    dim3 GridSize;
    GridSize.x = (n + BlockDim.x - 1) / BlockDim.x;
    GridSize.y = (n + BlockDim.y - 1) / BlockDim.y;

    TileMatMul <<<GridSize, BlockDim>>> (d_M, d_N, d_P, n);
    cudaDeviceSynchronize();

    // Transfer results from device to cpu
    cudaMemcpy(h_P, d_P, SizeInBytes, cudaMemcpyDeviceToHost);

    PrintResults(h_P, n);

    // Deallocate Memory 
    free(h_M); free(h_N); free(h_P);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);

    return 0;
}