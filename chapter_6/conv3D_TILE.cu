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
#include <stdio.h>
#include <stdlib.h>

/** @brief define constant parameters and constant memory inside the GPU */
#define RADIUS 1
#define KERNEL_SIZE (2 * RADIUS + 1)
#define KERNEL_VOL  (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE)
#define TILE_SIZE   8

__constant__ float F[KERNEL_VOL];
/*************************************************************************/

/**
 * @brief Implement tiled 3d conv with constant memory and shared memory
 * where input and output tiles are the same size.
 * @param In - input volume size (depth, height, width)
 * @param Out - output volume same size as input
 * 
 * @note Layout In[channel][row][col] flattened as: index = col + row * width + channel * width * height
*/
__global__
void TileConv3D(const float* __restrict__ In, 
                float* __restrict__ Out,
                int depth, int height, int width)
                {
                    // define and load data to shared mem
                    __shared__ float In_ds[TILE_SIZE][TILE_SIZE][TILE_SIZE];

                    int tx = threadIdx.x;
                    int ty = threadIdx.y;
                    int tz = threadIdx.z;

                    int col = tx + blockIdx.x * TILE_SIZE;
                    int row = ty + blockIdx.y * TILE_SIZE;
                    int channel = tz + blockIdx.z * TILE_SIZE;

                    if ((row < height) && (col < width) && (channel < depth))
                    {
                        In_ds[tz][ty][tx] = In[col + row * width + channel * width * height];
                    }
                    else
                    {
                        In_ds[tz][ty][tx] = 0.0f;
                    }

                    __syncthreads();

                    if ((row < height) && (col < width) && (channel < depth))
                    {
                        float Pval = 0.0f;
                        for (int fChannel=0; fChannel < KERNEL_SIZE; fChannel++)
                        {
                            for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
                            {
                                for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
                                {
                                    int inRow = ty - RADIUS + fRow;
                                    int inCol = tx - RADIUS + fCol;
                                    int inChannel = tz - RADIUS + fChannel;
                                    if ((inRow >=0) && (inRow < TILE_SIZE) &&
                                        (inCol >=0) && (inCol < TILE_SIZE) &&
                                        (inChannel >= 0) && (inChannel < TILE_SIZE))
                                        {
                                            Pval += F[fCol + fRow * KERNEL_SIZE + fChannel * KERNEL_SIZE * KERNEL_SIZE]
                                                    * In_ds[inChannel][inRow][inCol];
                                        }
                                    
                                    else
                                    {
                                        // check if it's ghost cells
                                        int gRow = row - RADIUS + fRow;
                                        int gCol = col - RADIUS + fCol;
                                        int gChannel = channel - RADIUS + fChannel;

                                        if ((gRow >=0) && (gRow < height) &&
                                            (gCol >=0) && (gCol < width) &&
                                            (gChannel >= 0) && (gChannel < depth))
                                            {
                                                Pval+= F[fCol + fRow * KERNEL_SIZE + fChannel * KERNEL_SIZE * KERNEL_SIZE]
                                                        * In[gCol + gRow * width + gChannel * width * height];
                                            }
                                    }
                                }
                            }
                        }

                        Out[col + row * width + channel * width * height] = Pval;
                    }

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
    dim3 block(TILE_SIZE, TILE_SIZE, TILE_SIZE);    // adjust as needed
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        (depth  + block.z - 1) / block.z
    );

    TileConv3D<<<grid, block>>>(d_in, d_out, depth, height, width);
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