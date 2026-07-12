#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>



// ========= Error checking =========
// Wrap any CUDA runtime call; aborts with file:line + cudaGetErrorString on failure.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)



// Define Parameters
constexpr int TILE_SIZE = 32;
constexpr int COARSE_FACTOR = 4;


__global__
void NaiveMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int M_Rows, const int M_Cols, const int N_Cols);


__global__
void TileMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int M_Rows, const int M_Cols, const int N_Cols);

__global__
void CoarseMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int M_Rows, const int M_Cols, const int N_Cols);
