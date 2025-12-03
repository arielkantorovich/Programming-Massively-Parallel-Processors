/**
 * @author Ariel Kantorovich
 * @details Tiled convolution algorithm that uses the same dimensions for input and output 
 * tiles and loads only internal elements of each tile into the shared memory. 
 * Note that the halo cells of an input tile of a block are also internal elements
 *  of neighbouring tiles. There is a possibility that while computing the values, 
 * the element at halo cells resides on L2 cache memory, hence saving us from the 
 * DRAM traffic. We can leave the halo cells in the original tiles rather than loading 
 * them to the shared memory.
*/

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>


/** @brief define constant memory and constant parameters */
#define RADIUS 1
#define KERNEL_SIZE (2 * RADIUS + 1)
#define TILE_SIZE 32

__constant__ float F[KERNEL_SIZE * KERNEL_SIZE];
/***********************************************************/


/**
 * @brief Implement Tile conv2D with constant memory and shared memory 
 * where input and output are the same size.
 * @param In - input matrix size (height, width)
 * @param Out - Output matrix size same as input 
*/
__global__
void TileConv2D(const float* __restrict__ In, float* __restrict__ Out, int width, int height)
{
    // Define and Load data to shared Memory
    __shared__ float In_ds[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;   int ty = threadIdx.y;
    int row = ty + blockIdx.y * TILE_SIZE;
    int col = tx + blockIdx.x * TILE_SIZE;

    if ((row < height) && (col < width))
    {
        In_ds[ty][tx] = In[col + row * width];
    }

    else
    {
        In_ds[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Calculate output files
    if ((row < height) && (col < width))
    {
        float Pval = 0.0f;
        for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
        {
            for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
            {
                // check if exist in L2-cache halo cells
                int inCol = threadIdx.x - RADIUS + fCol;
                int inRow = threadIdx.y - RADIUS + fRow;
                if ((inCol >= 0) && (inCol < TILE_SIZE) && (inRow >= 0) && (inRow < TILE_SIZE))
                {
                    Pval += F[fCol + fRow * KERNEL_SIZE] * In_ds[inRow][inCol];
                }

                else
                {
                    // Not exsist in cache need check if ghost cells or not and uploat from glob mem
                    int gCol = col - RADIUS + fCol;
                    int gRow = row - RADIUS + fRow;
                    if ((gCol >= 0) && (gCol < width) && (gRow >= 0) && (gRow < height))
                    {
                        Pval += F[fCol + fRow * KERNEL_SIZE] * In[gCol + gRow * width];
                    }
                }
            }
        }
        Out[col + row * width] = Pval;
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
    dim3 BlockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 GridSize;
    GridSize.x = (n + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (n + BlockSize.y - 1) / BlockSize.y;
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