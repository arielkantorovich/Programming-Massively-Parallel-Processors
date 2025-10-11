#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define TILE_SIZE 16

/**
 * General Matrix multiplication using Tile technique
 * @param M - input Matrix (rn, width)
 * @param N - input Matrix (width, cm)
 * @param P - Output Matrix (rn, cm)
 * @param (width, rm, cn) - matrices sizes parmaeters
*/
__global__
void GeneralMatMeul(float *M, float *N, float *P, int width, int rm, int cn)
{
    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
    __shared__ float Nds[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;           int ty = threadIdx.y;
    int bx = blockIdx.x;            int by = blockIdx.y;
    int Col = tx + TILE_SIZE*bx;    int Row = ty + TILE_SIZE*by;
    
    const int numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    float Pval = 0.0f;
    
    for (int ph=0; ph < numTiles; ph++)
    {
        const int aCol = ph * TILE_SIZE + tx;   // column in M
        const int bRow = ph * TILE_SIZE + ty;   // row in N

        Mds[ty][tx] = (Row < rm && aCol < width) ? M[Row * width + aCol] : 0.0f;
        Nds[ty][tx] = (bRow < width && Col < cn) ? N[Col + bRow * cn] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k=0; k < TILE_SIZE; k++)
        {
            Pval += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }

    if ((Row < rm) && (Col < cn))
    {
        P[cn * Row + Col] = Pval;
    }
}


int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Error: run ./exe <width> <rm> <cn> \n * <width> - is the width of matrix M \n * <rm> Row of matrix M. \n * <cn> - Colum of matrix N.\n");
        return -1;
    }

    int w = atoi(argv[1]);
    int rm = atoi(argv[2]);
    int cn = atoi(argv[3]);

    // Allocat Host Memory
    size_t Size_P = rm * cn * sizeof(float);
    size_t Size_M = rm * w * sizeof(float);
    size_t Size_N = cn * w * sizeof(float);

    float *h_P, *h_N, *h_M;

    h_P = (float *) malloc(Size_P);
    h_M = (float *) malloc(Size_M);
    h_N = (float *) malloc(Size_N);

    for (int i=0; i < rm*w; i++) h_M[i]=5.2f;
    for (int i=0; i < cn*w; i++) h_N[i]=2.2f;

    // Allocate Device Memory
    float *d_M, *d_P, *d_N;
    cudaMalloc((void **)&d_M, Size_M);
    cudaMalloc((void **)&d_P, Size_P);
    cudaMalloc((void **)&d_N, Size_N);

    // Transfer Data from Host to Device
    cudaMemcpy(d_M, h_M, Size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, Size_N, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 BlockDim(TILE_SIZE, TILE_SIZE);
    dim3 GridDim( (cn + TILE_SIZE - 1) / TILE_SIZE,
           (rm + TILE_SIZE - 1) / TILE_SIZE );
    
    GeneralMatMeul <<< GridDim, BlockDim>>> (d_M, d_N, d_P, w, rm, cn);
    cudaDeviceSynchronize();

    // Transfer results from Device To Host
    cudaMemcpy(h_P, d_P, Size_P, cudaMemcpyDeviceToHost);

    // Deallocate Memory
    free(h_M);  free(h_N); free(h_P);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);

    return 0;
}