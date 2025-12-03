/**
 * @author Ariel Kantorovich
 * @brief Implement convolutional 2D kernel using constant memory for the filter
 * to reduce DRAM access and using constant casch.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define RADIUS 1
#define KERNEL_SIZE (2 * RADIUS + 1)
__constant__ float F[KERNEL_SIZE * KERNEL_SIZE];


/**
 * @brief Conv2D device implementation using constant memory
 * @param In - input matrix size (height, width)
 * @param Out - output matrix same size as input
*/
__global__ 
void Conv2D(const float* __restrict__ In, float* __restrict__ Out, 
            unsigned int width, unsigned int height)
{
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;

    if (Row >= (int)height || Col >= (int)width)
        return;
    
    float Pval = 0.0f;

    for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
    {
        for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
        {
            int InRow = Row - RADIUS + fRow;
            int InCol = Col - RADIUS + fCol;
            if ((InRow >= 0) && (InRow < height) && (InCol >= 0) && (InCol < width))
            {
                Pval += F[fCol + fRow * KERNEL_SIZE] * In[InCol + InRow * width];
            }
        }
    }

    Out[Col + Row * width] = Pval;
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
    unsigned int NumThreads = 32;
    dim3 BlockSize(NumThreads, NumThreads, 1);
    dim3 GridSize;
    GridSize.x = (n + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (n + BlockSize.y - 1) / BlockSize.y;
    GridSize.z = 1;

    // Lanch kernel
    Conv2D <<< GridSize, BlockSize >>> (d_img, d_out, n, n);
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