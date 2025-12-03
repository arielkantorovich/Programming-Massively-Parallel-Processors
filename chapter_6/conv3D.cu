#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>



/** @brief define constant parameters and constant memory inside the GPU*/
#define RADIUS 1
#define KERNEL_SIZE (2 * RADIUS + 1)
#define KERNEL_VOL (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)

__constant__ float F[KERNEL_VOL];
/**************************************************************************/


/** @brief Implement conv3D using constant memory
 * @param In tensor with size (depth, height, width)
 * @param Out tensor with size same as input
*/
__global__
void conv3D(const float* __restrict__ In, float* __restrict__ Out, int depth, int width, int height)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = threadIdx.z + blockIdx.z * blockDim.z;

    if ((row >= height) || (col >= width) || (channel >= depth))
        return;
    
    float Pval = 0.0f;
    for (int fChannel=0; fChannel < KERNEL_SIZE; fChannel++)
    {
        for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
        {
            for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
            {
                int inRow = row - RADIUS + fRow;
                int inCol = col - RADIUS + fCol;
                int inChannel = channel - RADIUS + fChannel;
                if ((inRow >= 0) && (inRow < height) &&
                    (inCol >= 0) && (inCol < width) &&
                    (inChannel >=0) && (inChannel < depth))
                    {
                        Pval += F[fCol + fRow * KERNEL_SIZE + fChannel * KERNEL_SIZE * KERNEL_SIZE] 
                                * In[inCol + inRow * width + inChannel * width * height];

                    }

            }
        }
    }
    Out[col + row * width + channel * width * height] = Pval;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Usage: ./exe <depth> <height> <width>\n");
        return -1;
    }

    int depth  = atoi(argv[1]);
    int height = atoi(argv[2]);
    int width  = atoi(argv[3]);

    size_t volumeSize = (size_t)depth * height * width;
    size_t bytes = volumeSize * sizeof(float);

    // -------------------------
    // Allocate host memory
    // -------------------------
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input volume
    for (size_t i = 0; i < volumeSize; i++)
        h_in[i] = 1.0f;     // example constant value

    // -------------------------
    // Create 3D filter (3×3×3)
    // -------------------------
    float h_kernel[KERNEL_VOL];
    for (int i = 0; i < KERNEL_VOL; i++)
        h_kernel[i] = 1.0f;   // simple all-ones kernel

    // Copy kernel to GPU constant memory
    cudaMemcpyToSymbol(F, h_kernel, sizeof(float) * KERNEL_VOL);

    // -------------------------
    // Allocate GPU memory
    // -------------------------
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  bytes);
    cudaMalloc((void**)&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // -------------------------
    // Launch kernel
    // -------------------------
    dim3 block(8, 8, 8);    // adjust as needed
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        (depth  + block.z - 1) / block.z
    );

    conv3D<<<grid, block>>>(d_in, d_out, depth, width, height);
    cudaDeviceSynchronize();

    // -------------------------
    // Copy result back
    // -------------------------
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // -------------------------
    // Print some elements
    // -------------------------
    printf("Example output values:\n");
    for (int z = 0; z < min(depth, 2); ++z)
    {
        for (int y = 0; y < min(height, 2); ++y)
        {
            for (int x = 0; x < min(width, 4); ++x)
            {
                int idx = z * (height * width) + y * width + x;
                printf("Out[%d][%d][%d] = %f\n", z, y, x, h_out[idx]);
            }
        }
    }

    // -------------------------
    // Free resources
    // -------------------------
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}