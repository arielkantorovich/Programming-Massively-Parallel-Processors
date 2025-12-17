#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "param_c.hpp"

#define IN_TILE_DIM 8
#define RADIUS 1
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * RADIUS)

/**
 * @brief implement stencil kernel with shared memory technique
 * @param in - input 3D size (n, n, n)
 * @param out - output 3D size (n, n ,n)
 * @param N - size in each axes
*/
__global__ void stencil_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
{
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    // define and load to shared memory
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >=0 && i < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[k + j * N + i * N * N];
    }
    __syncthreads();

    // Calculate stencil from shared memory
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1)
    {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 
            && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 
            && threadIdx.x >= 1 && threadIdx.z < IN_TILE_DIM-1)
            {
                out[k + j * N + i * N * N] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] + 
                                c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] + 
                                c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] + 
                                c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] + 
                                c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
            }
    }
}