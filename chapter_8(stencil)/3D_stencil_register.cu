#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "param_c.hpp"

#define IN_TILE 32
#define RADIUS 1 // order of the stencil
#define OUT_TILE (IN_TILE - 2 * RADIUS)

/**
 * @brief 3D stencil kernel using register tiling in the z-direction
 *        (PMPP Fig. 8.12).
 *
 *        Only the CURRENT z-plane is kept in shared memory, because the
 *        stencil reads its x/y neighbours (which belong to other threads).
 *        The PREVIOUS and NEXT planes are read only at the thread's own
 *        (x, y) position, so they are held in thread-private REGISTERS
 *        instead of shared memory. This cuts shared-memory usage to a third
 *        and removes two thirds of the shared-memory traffic.
 *
 * @param In     - input  3D volume (depth, height, width), row-major
 * @param Out    - output 3D volume (depth, height, width), row-major
 * @param width  - size along x
 * @param height - size along y
 * @param depth  - size along z
 */
__global__ void register3DStencil(const float* __restrict__ In,
                                   float* __restrict__ Out,
                                   const int width,
                                   const int height,
                                   const int depth)
{
    int tx = threadIdx.x;   int ty = threadIdx.y;

    int iStart = blockIdx.z * OUT_TILE;
    int j = ty + blockIdx.y * OUT_TILE - RADIUS;
    int i = tx + blockIdx.x * OUT_TILE - RADIUS;

    // Only the current plane lives in shared memory.
    __shared__ float in_Curr_s[IN_TILE][IN_TILE];

    // Previous / next planes are kept in registers (own element only).
    float in_Prev;
    float in_Next;

    // Preload the previous plane (iStart - 1) into a register.
    if ((iStart - 1 >= 0) && (iStart - 1 < depth) &&
        (j >= 0) && (j < height) && (i >= 0) && (i < width))
    {
        in_Prev = In[(iStart - 1) * height * width + j * width + i];
    }

    // Preload the current plane (iStart) into shared memory.
    if ((iStart >= 0) && (iStart < depth) &&
        (j >= 0) && (j < height) && (i >= 0) && (i < width))
    {
        in_Curr_s[ty][tx] = In[iStart * height * width + j * width + i];
    }

    // March through OUT_TILE planes along z.
    for (int k = iStart; k < iStart + OUT_TILE; k++)
    {
        // Load the next plane (k + 1) into a register.
        if ((k + 1 >= 0) && (k + 1 < depth) &&
            (j >= 0) && (j < height) && (i >= 0) && (i < width))
        {
            in_Next = In[(k + 1) * height * width + j * width + i];
        }
        __syncthreads();

        // Compute output for plane k. Only interior global voxels are written,
        // and only interior (non-halo) threads compute so that the x/y
        // neighbour reads stay inside shared memory.
        if ((k >= 1) && (k < depth - 1) &&
            (j >= 1) && (j < height - 1) &&
            (i >= 1) && (i < width - 1))
        {
            if ((ty >= 1) && (ty < IN_TILE - 1) &&
                (tx >= 1) && (tx < IN_TILE - 1))
            {
                Out[k * width * height + j * width + i] =
                      c0 * in_Curr_s[ty][tx]        // center
                    + c1 * in_Curr_s[ty][tx - 1]    // -x   (shared)
                    + c2 * in_Curr_s[ty][tx + 1]    // +x   (shared)
                    + c3 * in_Curr_s[ty - 1][tx]    // -y   (shared)
                    + c4 * in_Curr_s[ty + 1][tx]    // +y   (shared)
                    + c5 * in_Prev                  // -z   (register)
                    + c6 * in_Next;                 // +z   (register)
            }
        }
        __syncthreads();

        // Roll the window forward: prev <- curr <- next.
        in_Prev = in_Curr_s[ty][tx];  // read curr before it is overwritten
        __syncthreads();
        in_Curr_s[ty][tx] = in_Next;  // promote next plane into shared memory
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

    // Block is 2D (IN_TILE x IN_TILE); z is handled by register tiling.
    dim3 BlockSize(IN_TILE, IN_TILE, 1);
    dim3 GridSize;
    GridSize.x = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.y = (n + OUT_TILE - 1) / OUT_TILE;
    GridSize.z = (n + OUT_TILE - 1) / OUT_TILE;

    // Launch kernel (width = height = depth = n)
    register3DStencil<<<GridSize, BlockSize>>>(d_in, d_out, n, n, n);

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
