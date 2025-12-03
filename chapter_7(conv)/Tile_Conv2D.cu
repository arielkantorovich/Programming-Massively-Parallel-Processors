#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

/** @brief define constant memory and constant parameters*/
#define RADIUS   1
#define KERNEL_SIZE (2 * RADIUS + 1)
#define IN_TILE  32
#define OUT_TILE (IN_TILE - 2 * RADIUS)

__constant__ float F[KERNEL_SIZE * KERNEL_SIZE];
/*********************************************************/



/**
 * @brief Implement 2D conv using tile technique and constant Device memory
 * @param In - input matrix size (height, width) 
 * @param Out - output matrix same size as inout.
 * @note Input tile and output Tile are in diffrent size
*/
__global__
void TileConv2D(const float* __restrict__ In, float* __restrict__ Out, int width, int height)
{
    // Define and Load data to shared memory
    __shared__ float In_ds[IN_TILE][IN_TILE];

    int ty = threadIdx.y;   int tx = threadIdx.x;
    int row = ty + blockIdx.y * OUT_TILE - RADIUS;
    int col = tx + blockIdx.x * OUT_TILE - RADIUS;
    
    if ((row >= 0) && (row < height) && (col >= 0) && (col < width))
    {
        In_ds[ty][tx] = In[col + row * width];
    }
    else
    {
        In_ds[ty][tx] = 0.0f;
    }

    __syncthreads();
    
    // Load from shared memory and calculate results
    int tileRow = ty - RADIUS;
    int tileCol = tx - RADIUS;
    if ((row >= 0) && (row < height) && (col >= 0) && (col < width))
    {
        if ((tileCol >= 0) && (tileCol < OUT_TILE) && 
            (tileRow >= 0) && (tileRow < OUT_TILE))
        {
            float Pval = 0.0f;
            for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
            {
                for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
                {
                    Pval += F[fCol + fRow * KERNEL_SIZE] * In_ds[tileRow + fRow][tileCol + fCol];
                }
            }
            Out[col + row * width] = Pval;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc !=2)
    {
        printf("Error: run ./exe <n> matrix width. \n");
        return -1;
    }

    int n = atoi(argv[1]);
    const float kernel[] = {
                        1.0f, 2.0f, 1.0f,
                        2.0f, 4.0f, 2.0f,
                        1.0f, 2.0f, 1.0f
                        };
    
    size_t SizeInBytes = n * n * sizeof(float);
    size_t KERNEL_SizeInBytes = sizeof(float) * KERNEL_SIZE * KERNEL_SIZE;
    
    float *Img, *Out;
    Img = (float *)malloc(SizeInBytes);
    Out = (float *)malloc(SizeInBytes);

    for (int i = 0; i < n*n; i++)
    {
        Img[i] = 3.3f;
    }

    // Allocate memory in Gpu
    float *d_img, *d_out;
    cudaMalloc((void **)&d_img, SizeInBytes);
    cudaMalloc((void **)&d_out, SizeInBytes);
    
    /** @brief Allocate constant memory*/
    cudaMemcpyToSymbol(F, kernel, KERNEL_SizeInBytes);

    // Memory transfer
    cudaMemcpy(d_img, Img, SizeInBytes, cudaMemcpyHostToDevice);

    // Define Grid
    dim3 BlockSize(IN_TILE, IN_TILE, 1);
    dim3 GridSize;
    GridSize.x = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.y = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.z = 1;

    // Lanch kernel
    TileConv2D <<< GridSize, BlockSize >>> (d_img, d_out, n, n);
    cudaDeviceSynchronize();

    // Return resutls to host
    cudaMemcpy(Out, d_out, SizeInBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("Out[%i][%i] = %f \n", i, j, Out[j + i * n]);
        }
    }

    // release memory
    free(Img); free(Out);
    cudaFree(d_img); cudaFree(d_out);

    return 0;
}