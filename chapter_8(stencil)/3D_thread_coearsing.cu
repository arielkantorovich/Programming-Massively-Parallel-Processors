#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "param_c.hpp"

#define IN_TILE 32
#define RADIUS 1 // which is the order of the stencil calculation
#define OUT_TILE (IN_TILE - 2 * RADIUS)

/**
 * @brief 3D stencil kernel using thread coarsening in the z-direction.
 *        Each block keeps only three z-planes (prev/curr/next) in shared
 *        memory and marches through OUT_TILE planes along z (PMPP Fig. 8.10).
 * @param In    - input  3D volume (depth, height, width), row-major
 * @param Out   - output 3D volume (depth, height, width), row-major
 * @param width  - size along x
 * @param height - size along y
 * @param depth  - size along z
 */
__global__ void coaresing3DStencil(const float* __restrict__ In,
                                    float* __restrict__ Out,
                                    const int width,
                                    const int height,
                                    const int depth)
{
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // iStart: first z-plane this block is responsible for (no halo in z,
    // planes are streamed one at a time through the register/shmem window).
    int iStart = blockIdx.z * OUT_TILE;
    int j = ty + blockIdx.y * OUT_TILE - RADIUS;
    int i = tx + blockIdx.x * OUT_TILE - RADIUS;

    __shared__ float in_Curr[IN_TILE][IN_TILE];
    __shared__ float in_Next[IN_TILE][IN_TILE];
    __shared__ float in_Prev[IN_TILE][IN_TILE];

    // Load the previous plane (iStart - 1)
    if ((iStart - 1 >= 0) && (iStart - 1 < depth) &&
        (j >= 0) && (j < height) && (i >= 0) && (i < width))
    {
        in_Prev[ty][tx] = In[(iStart - 1) * height * width + j * width + i];
    }

    // Load the current plane (iStart)
    if ((iStart >= 0) && (iStart < depth) &&
        (j >= 0) && (j < height) && (i >= 0) && (i < width))
    {
        in_Curr[ty][tx] = In[iStart * height * width + j * width + i];
    }

    // March through OUT_TILE planes along z
    for (int k = iStart; k < iStart + OUT_TILE; k++)
    {
        // Load the next plane (k + 1)
        if ((k + 1 >= 0) && (k + 1 < depth) &&
            (j >= 0) && (j < height) && (i >= 0) && (i < width))
        {
            in_Next[ty][tx] = In[(k + 1) * height * width + j * width + i];
        }
        __syncthreads();

        // Compute output for plane k. Only interior global voxels are written,
        // and only interior (non-halo) threads perform the computation so that
        // neighbour accesses stay inside shared memory.
        if ((k >= 1) && (k < depth - 1) &&
            (j >= 1) && (j < height - 1) &&
            (i >= 1) && (i < width - 1))
        {
            if ((ty >= 1) && (ty < IN_TILE - 1) &&
                (tx >= 1) && (tx < IN_TILE - 1))
            {
                Out[k * width * height + j * width + i] =
                      c0 * in_Curr[ty][tx]
                    + c1 * in_Curr[ty][tx - 1]
                    + c2 * in_Curr[ty][tx + 1]
                    + c3 * in_Curr[ty - 1][tx]
                    + c4 * in_Curr[ty + 1][tx]
                    + c5 * in_Prev[ty][tx]
                    + c6 * in_Next[ty][tx];
            }
        }
        __syncthreads();

        // Roll the register/shmem window forward: prev <- curr <- next
        in_Prev[ty][tx] = in_Curr[ty][tx];
        in_Curr[ty][tx] = in_Next[ty][tx];
    }
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error need run ./<exe file> <n> where <n> is the size in each 3D axes. \n");
        return -1;
    }

    int n = atoi(argv[1]);
    if (n <= 0)
    {
        printf("Error: <n> must be a positive integer. \n");
        return -1;
    }

    size_t SizeInBytes = (size_t)n * n * n * sizeof(float);

    // Allocate host (page-locked) memory
    float *h_in, *h_out;
    cudaMallocHost((void**)&h_in, SizeInBytes);
    cudaMallocHost((void**)&h_out, SizeInBytes);
    putConst(h_in, 2.3f, n);

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, SizeInBytes);
    cudaMalloc((void**)&d_out, SizeInBytes);

    // Transfer input from host to device
    cudaMemcpy(d_in, h_in, SizeInBytes, cudaMemcpyHostToDevice);

    // Block is 2D (IN_TILE x IN_TILE); coarsening runs along z.
    dim3 BlockSize(IN_TILE, IN_TILE, 1);
    dim3 GridSize;
    GridSize.x = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.y = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.z = (n + OUT_TILE - 1) / OUT_TILE;

    // Launch kernel (width = height = depth = n)
    coaresing3DStencil<<<GridSize, BlockSize>>>(d_in, d_out, n, n, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s \n", cudaGetErrorString(err));
        cudaFree(d_in); cudaFree(d_out);
        cudaFreeHost(h_in); cudaFreeHost(h_out);
        return -1;
    }
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, SizeInBytes, cudaMemcpyDeviceToHost);

    // Print results
    PrintResult(h_out, n);

    // Free memory
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in); cudaFreeHost(h_out);

    return 0;
}
