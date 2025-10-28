#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define TILE 16

/**
 * @brief Tile matrix multiplication corner turning technique.
 * @param A - input matrix size (m x k)
 * @param B - input matrix size (k x n)
 * @param C - output matrix size (m x n)
*/
__global__ void gemm_corner_turning(const float* __restrict__ A, // (M x K) row-major
                                    const float* __restrict__ B, // (K x N) column-major
                                    float* __restrict__ C,       // (M x N) row-major
                                    int M, int K, int N)
{
    __shared__ float As[TILE][TILE];       // tile of A in shared
    __shared__ float Bs[TILE][TILE + 1];   // tile of B in shared (transposed, +1 to ease bank conflicts)

    const int tx = threadIdx.x;                 // 0..TILE-1 (varies fastest inside a warp)
    const int ty = threadIdx.y;                 // 0..TILE-1
    const int row = blockIdx.y * TILE + ty;     // row in C output
    const int col = blockIdx.x * TILE + tx;     // col in C output

    float acc = 0.0f;                           // accumulator for C[row,col]
    const int phases = (K + TILE - 1) / TILE;   // how many K-tiles to traverse

    for (int ph = 0; ph < phases; ++ph) {
        // ---- LOAD A TILE (row-major) ----
        // Row-major linear index: aIdx = aRow*K + aCol
        const int aRow = row;
        const int aCol = ph * TILE + tx;              // tx walks columns → coalesced for A
        As[ty][tx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        // ---- LOAD B TILE with CORNER TURNING (B is column-major) ----
        // Column-major linear index: bIdx = bRow + bCol*K  (leading dim = K)
        // We want warp lanes (tx) to walk ROWS (contiguous in column-major):
        const int bRow = ph * TILE + tx;              // tx walks rows → coalesced for B
        const int bCol = blockIdx.x * TILE + ty;      // ty stays constant within a warp → one column per warp
        float bval = (bRow < K && bCol < N) ? B[bRow + bCol * K] : 0.0f;

        // Store TRANSPOSED into shared so compute loop can read Bs[k][tx]
        Bs[tx][ty] = bval;

        __syncthreads();

        // ---- COMPUTE this tile’s contribution ----
        // Reads: As[ty][k] (row ty across columns), Bs[k][tx] (row k across columns) — fast from shared
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];

        __syncthreads(); // safe reuse of shared in next phase
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

int main(int argc, char **argv)
{
    return 0;
}