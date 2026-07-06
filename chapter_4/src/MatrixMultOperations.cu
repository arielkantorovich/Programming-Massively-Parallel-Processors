#include "MatrixMultOperations.cuh"


/**
 * @brief Implement naive matrix multiplication each thread calculate one output 
 * @param P output matrix size M_RowsxN_Cols.
 * @param M input matrix size M_RowsxM_Cols.
 * @param N input matrix size N_RowsxN_Cols.
 * @param size of out matrix
 */
__global__
void NaiveMatMul_kernel(float* __restrict__ P, 
                        const float* __restrict__ M, 
                        const float* __restrict__ N,
                        const int M_Rows, const int M_Cols, const int N_Cols)
                        {
                            int Col = threadIdx.x + blockIdx.x * blockDim.x;
                            int Row = threadIdx.y + blockIdx.y * blockDim.y;
                            if ((Row < M_Rows) && (Col < N_Cols))
                            {
                                float Pval=0;
                                for (int i=0; i < M_Cols; ++i)
                                {
                                    Pval+= M[Row * M_Cols + i] * N[Col + i * N_Cols];
                                }
                                P[Row * N_Cols + Col] = Pval;
                            }
                        }




/**
 * @brief Implement matrix multiplication with tile teachnique
 * @param P output matrix nxn.
 * @param M input matrix nxn.
 * @param N input matrix nxn.
 * @param n matrix sizr
 * @note assume sqaure matrix, assume n / TileSize is integr, assume TILE_SIZE=BLOCK_SIZE.
 */
__global__
void NaiveTileMatMul_kernel(float* __restrict__ P, 
                    const float* __restrict__ M, 
                    const float* __restrict__ N, 
                    const int n)
                    {
                        int tx = threadIdx.x;   int ty = threadIdx.y;
                        int bx = blockIdx.x;    int by = blockIdx.y;
                        
                        int Col = tx + bx * TILE_SIZE;  
                        int Row = ty + by * TILE_SIZE;

                        // allocate shared Memory
                        __shared__ float Mds[TILE_SIZE][TILE_SIZE];
                        __shared__ float Nds[TILE_SIZE][TILE_SIZE];
                        
                        float Pval = 0.0f;

                        for (int ph=0; ph < n/TILE_SIZE; ph++)
                        {
                            // Load data to shared memory
                            Mds[ty][tx] = M[(tx + ph*TILE_SIZE) + Row*n];
                            Nds[ty][tx] = N[Col + (ph*TILE_SIZE + ty) * n];
                            __syncthreads();

                            // Calculate multiplication between the shreadMem
                            for (int k=0; k < TILE_SIZE; ++k)
                            {
                                Pval += Mds[ty][k] * Nds[k][tx];
                            }

                            __syncthreads();
                        }

                        P[Col + Row * n] = Pval;
                    }


/**
 * @brief Matrix Multiplication Bounds check.
 * @param P output matrix rmxcn.
 * @param M input matrix rmxw.
 * @param N input matrix wxcn.
 * @param (rm, w, cn) size of matrices
 */
__global__
void GeneralTileMatMul_kernel(float* __restrict__ P, 
                    const float* __restrict__ M, 
                    const float* __restrict__ N, 
                    const int rm, const int w, const int cn)
                    {
                        __shared__ float Mds[TILE_SIZE][TILE_SIZE];
                        __shared__ float Nds[TILE_SIZE][TILE_SIZE];

                        int tx = threadIdx.x;   int bx = blockIdx.x;
                        int ty = threadIdx.y;   int by = blockIdx.y;
                        
                        int Col = tx + bx * TILE_SIZE;
                        int Row = ty + by * TILE_SIZE;
                        
                        int numTiles = (w - 1 + TILE_SIZE) / TILE_SIZE;
                        float Pval = 0.0f;

                        for (int ph=0; ph < numTiles; ++ph)
                        {
                            int Mcol = tx + ph * TILE_SIZE;
                            int Nrow = ty + ph * TILE_SIZE;
                            // Load data to shared memory
                            Mds[ty][tx] = (Row < rm && Mcol < w) ? M[Mcol + Row * w] : 0.0f;
                            Nds[ty][tx] = (Nrow < w && Col < cn) ? N[Col + Nrow * cn] : 0.0f;
                            __syncthreads();
                            // Calculate multiplication
                            for (int k=0; k < TILE_SIZE; ++k)
                            {
                                Pval += Mds[ty][k] * Nds[k][tx];
                            }
                            __syncthreads();
                        }

                        if (Row < rm && Col < cn)
                        {
                            P[Col + Row * cn] = Pval;
                        }
                    }


// ===================================================================
// Host-side launchers: allocate device memory, copy data in, launch
// the matching kernel, copy the result back, and free device memory.
// ===================================================================

void NaiveMatMulDevice(const float* h_M, const float* h_N, float* h_P,
                        int M_Rows, int M_Cols, int N_Cols)
{
    const size_t sizeM = sizeof(float) * M_Rows * M_Cols;
    const size_t sizeN = sizeof(float) * M_Cols * N_Cols;
    const size_t sizeP = sizeof(float) * M_Rows * N_Cols;

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, sizeM));
    CUDA_CHECK(cudaMalloc((void**)&d_N, sizeN));
    CUDA_CHECK(cudaMalloc((void**)&d_P, sizeP));

    CUDA_CHECK(cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N_Cols + block.x - 1) / block.x, (M_Rows + block.y - 1) / block.y);
    NaiveMatMul_kernel<<<grid, block>>>(d_P, d_M, d_N, M_Rows, M_Cols, N_Cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost));

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

void NaiveTileMatMulDevice(const float* h_M, const float* h_N, float* h_P, int n)
{
    const size_t sizeBytes = sizeof(float) * n * n;

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_N, sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_P, sizeBytes));

    CUDA_CHECK(cudaMemcpy(d_M, h_M, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, sizeBytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(n / TILE_SIZE, n / TILE_SIZE);
    NaiveTileMatMul_kernel<<<grid, block>>>(d_P, d_M, d_N, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_P, d_P, sizeBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

void GeneralTileMatMulDevice(const float* h_M, const float* h_N, float* h_P,
                              int rm, int w, int cn)
{
    const size_t sizeM = sizeof(float) * rm * w;
    const size_t sizeN = sizeof(float) * w * cn;
    const size_t sizeP = sizeof(float) * rm * cn;

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc((void**)&d_M, sizeM));
    CUDA_CHECK(cudaMalloc((void**)&d_N, sizeN));
    CUDA_CHECK(cudaMalloc((void**)&d_P, sizeP));

    CUDA_CHECK(cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cn + TILE_SIZE - 1) / TILE_SIZE, (rm + TILE_SIZE - 1) / TILE_SIZE);
    GeneralTileMatMul_kernel<<<grid, block>>>(d_P, d_M, d_N, rm, w, cn);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost));

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}