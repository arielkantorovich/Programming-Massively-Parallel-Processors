#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Define Parameters
constexpr int TILE_SIZE = 16;

// ========= Error checking =========
// Wrap any CUDA runtime call; aborts with file:line + cudaGetErrorString on failure.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

__global__
void NaiveMatMul_kernel(float* __restrict__ P, 
                        const float* __restrict__ M, 
                        const float* __restrict__ N,
                        const int M_Rows, const int M_Cols, const int N_Cols);


__global__
void NaiveTileMatMul_kernel(float* __restrict__ P, 
                    const float* __restrict__ M, 
                    const float* __restrict__ N, 
                    const int n);

__global__
void GeneralTileMatMul_kernel(float* __restrict__ P, 
                    const float* __restrict__ M, 
                    const float* __restrict__ N, 
                    const int rm, const int w, const int cn);

// ========= Host-side launchers (malloc / copy / launch / copy / free) =========
// Each takes plain host pointers so main.cpp never has to touch <<<...>>> syntax.

/**
 * @brief Host launcher for NaiveMatMul_kernel.
 * @param h_M input matrix M_Rows x M_Cols (row-major).
 * @param h_N input matrix M_Cols x N_Cols (row-major); N's row count must equal M_Cols.
 * @param h_P output matrix M_Rows x N_Cols (row-major), pre-allocated by caller.
 * @param M_Rows,M_Cols,N_Cols matrix dimensions.
 */
void NaiveMatMulDevice(const float* h_M, const float* h_N, float* h_P,
                        int M_Rows, int M_Cols, int N_Cols);

/**
 * @brief Host launcher for NaiveTileMatMul_kernel.
 * @param h_M,h_N input square matrices n x n (row-major).
 * @param h_P output matrix n x n (row-major), pre-allocated by caller.
 * @param n matrix size; must be an exact multiple of TILE_SIZE (kernel assumption).
 */
void NaiveTileMatMulDevice(const float* h_M, const float* h_N, float* h_P, int n);

/**
 * @brief Host launcher for GeneralTileMatMul_kernel.
 * @param h_M input matrix rm x w (row-major).
 * @param h_N input matrix w x cn (row-major).
 * @param h_P output matrix rm x cn (row-major), pre-allocated by caller.
 * @param rm,w,cn matrix dimensions; no divisibility/shape assumptions.
 */
void GeneralTileMatMulDevice(const float* h_M, const float* h_N, float* h_P,
                              int rm, int w, int cn);