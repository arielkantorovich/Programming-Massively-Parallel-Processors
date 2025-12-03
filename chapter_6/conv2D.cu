#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Implement 2D-convolution on GPU device, with padding zero
 * @param In - input matrix size (height, width)
 * @param F - filter matrix size (2r+1, 2r+1)
 * @param Out - output matrix size (height, width)
 * @param r - filter radius
 * @param width
 * @param height
 * 
*/
__global__
void DeviceConv2D(const float* __restrict__ In, const float* __restrict__ F, float* __restrict__ Out, 
                unsigned int r, unsigned int width, unsigned int height)
{
    const int KERNEL_SIZE = 2 * r + 1;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;

    // Skip threads outside the image
    if (Row >= (int)height || Col >= (int)width)
        return;
    
    float Pval = 0.0f;

    for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
    {
        for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
        {
            int InRow = Row - r + fRow;
            int InCol = Col - r + fCol;
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
    int r = 1;
    const float kernel[] = {
                        1.0f, 2.0f, 1.0f,
                        2.0f, 4.0f, 2.0f,
                        1.0f, 2.0f, 1.0f
                        };
    
    size_t SizeInBytes = n * n * sizeof(float);
    size_t KERNEL_SizeInBytes = sizeof(float) * (2 * r + 1) * (2 * r + 1);
    
    float *Img, *Out;
    Img = (float *)malloc(SizeInBytes);
    Out = (float *)malloc(SizeInBytes);

    for (int i = 0; i < n*n; i++)
    {
        Img[i] = 3.3f;
    }

    // Allocate memory in Gpu
    float *d_img, *d_out, *d_F;
    cudaMalloc((void **)&d_img, SizeInBytes);
    cudaMalloc((void **)&d_out, SizeInBytes);
    cudaMalloc((void **)&d_F, KERNEL_SizeInBytes);

    // Memory transfer
    cudaMemcpy(d_F, kernel, KERNEL_SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img, Img, SizeInBytes, cudaMemcpyHostToDevice);

    // Define Grid
    unsigned int NumThreads = 32;
    dim3 BlockSize(NumThreads, NumThreads, 1);
    dim3 GridSize;
    GridSize.x = (n + BlockSize.x - 1) / BlockSize.x;
    GridSize.y = (n + BlockSize.y - 1) / BlockSize.y;
    GridSize.z = 1;

    // Lanch kernel
    DeviceConv2D <<< GridSize, BlockSize >>> (d_img, d_F, d_out, r, n, n);
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
    cudaFree(d_img); cudaFree(d_out); cudaFree(d_F);

    return 0;
}