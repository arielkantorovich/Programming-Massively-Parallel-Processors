#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

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


__global__
void TileMatMul(float *A, float *B, float *C, int w)
{
    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];
    unsigned int numTiles = (w + TILE_SIZE - 1) / TILE_SIZE;

    int tx = threadIdx.x;   int ty = threadIdx.y;
    int Col = tx + TILE_SIZE * blockIdx.x;
    int Row = ty + TILE_SIZE * blockIdx.y;

    float c_value = 0.0f;
    for (int ph=0; ph < numTiles; ph++)
    {
        // first load data to shared memory
        Ads[ty][tx] = A[(tx + ph * TILE_SIZE) + Row * w];
        Bds[ty][tx] = B[Col + w * (ph * TILE_SIZE + ty)];
        __syncthreads();

        // calculate multiplcation on tile (shmem)
        for (int k = 0; k < TILE_SIZE; k++)
        {
            c_value += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[Col + Row * w] = c_value;
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error: ./<exe> <n> where <n> is matrix size \n");
    }
    
    int n = atoi(argv[1]);
    size_t SizeInBytes = n * n * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(SizeInBytes);
    h_B = (float*)malloc(SizeInBytes);
    h_C = (float*)malloc(SizeInBytes);

    SetConstant(h_A, n, 2.0f);
    SetConstant(h_B, n, 0.3f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, SizeInBytes);
    cudaMalloc((void**)&d_B, SizeInBytes);
    cudaMalloc((void**)&d_C, SizeInBytes);

    // Define Grid & kernel size
    dim3 BlockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 GridSize;
    GridSize.x = (n + TILE_SIZE - 1) / TILE_SIZE;
    GridSize.y = (n + TILE_SIZE - 1) / TILE_SIZE;
    GridSize.z = 1;

    // Transfer data from host to device 
    cudaMemcpy(d_A, h_A, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    TileMatMul <<< GridSize, BlockSize >>> (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Return results to host
    cudaMemcpy(h_C, d_C, SizeInBytes, cudaMemcpyDeviceToHost);

    // Release Memo
    free(h_A);  free(h_B); free(h_C);
    cudaFree(d_A);  cudaFree(d_B);  cudaFree(d_C);

    return 0;
}
