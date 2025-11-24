#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define COARSE_FACTOR 4


/**
 * @brief Matrix multiplication implementation using tile technique
 * and thread coarsening.
 * @param M - input matrix size (I, I)
 * @param N - input matrix size (I, I)
 * @param P - output matrix result (I, I)
 * @param width - col of matrix M, row of N
 * @note for simplicy, implementation assume sqaure matrices
*/
__global__
void TileMatMulCoars(float *M, float *N, float *P, int width)
{
    const int numTiles = (width - 1 + TILE_SIZE) / TILE_SIZE;
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;         int ty = threadIdx.y;
    int bx = blockIdx.x;          int by = blockIdx.y;
    
    int Row = ty + by * TILE_SIZE;
    int Col_start = tx + bx * TILE_SIZE * COARSE_FACTOR;

    float Pval[COARSE_FACTOR];
    for  (int c=0; c < COARSE_FACTOR; c++)
    {
        Pval[c] = 0.0f;
    }
    
    for (int ph = 0; ph < numTiles; ph++)
    {
        Mds[ty][tx] = M[Row * width + ph * TILE_SIZE + tx];
        
        for (int c=0; c < COARSE_FACTOR; c++)
        {
            int Col = Col_start + c * TILE_SIZE;
            Nds[ty][tx] = N[Col + width * (ph * TILE_SIZE + ty)] ;
            __syncthreads();

            for (int k = 0; k < TILE_SIZE; k++)
            {
                Pval[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c=0; c < COARSE_FACTOR; c++)
    {
        int Col = Col_start + c * TILE_SIZE;
        P[Col + Row * width] = P[c];
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error: need run ./exe <n> \n where <n> is matrix size (sqaure matrix. \n)");
        return -1;
    }
    
    int n = atoi(argv[1]);
    size_t SizeBytes = n * n * sizeof(float);

    // Allocate cpu memory
    float *h_P, *h_M, *h_N;
    h_P = (float *)malloc(SizeBytes);
    h_M = (float *)malloc(SizeBytes);
    h_N = (float *)malloc(SizeBytes);

    for (int i=0; i < n*n; i++)
    {
        h_M[i] = 2.0f;
        h_N[i] = 3.3f;
    }

    // Allocate device memory
    float *d_P, *d_M, *d_N;
    cudaMalloc((void **) &d_M, SizeBytes);
    cudaMalloc((void **) &d_N, SizeBytes);
    cudaMalloc((void **) &d_P, SizeBytes);

    // Transfer Data from Host to Device
    cudaMemcpy(d_M, h_M, SizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, SizeBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 BlockDim(TILE_SIZE, TILE_SIZE);
    dim3 GridDim;
    GridDim.x = (n + TILE_SIZE - 1) / TILE_SIZE;
    GridDim.y = (n + TILE_SIZE - 1) / TILE_SIZE;

    TileMatMulCoars <<< GridDim, BlockDim >>> (d_M, d_N, d_P, n);
    cudaDeviceSynchronize();

    // Return results to Host
    cudaMemcpy(h_P, d_P, SizeBytes, cudaMemcpyDeviceToHost);

    // Deallocate Memory 
    free(h_M); free(h_N); free(h_P);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);

    return 0;
}