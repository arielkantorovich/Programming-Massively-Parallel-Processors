#include "GEMM.cuh"



/**
 * @brief Naive matrix multiplication P=MN.
 * @param P output matrix size M_RowsxN_Cols.
 * @param M input matrix size M_RowsxM_Cols.
 * @param N input matrix size M_ColsxN_Cols. (M_Cols=N_Rows)
 * @note naive implementation without tile technique or thread coarsing. 
 */
__global__
void NaiveMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int M_Rows, const int M_Cols, const int N_Cols)
                               {
                                    int Rows = threadIdx.y + blockIdx.y * blockDim.y;
                                    int Cols = threadIdx.x + blockIdx.x * blockDim.x;
                                    if ((Rows < M_Rows) && ( Cols < N_Cols))
                                    {
                                        float sum = 0.0f;
                                        for (int k=0; k < M_Cols; k++)
                                        {
                                            sum += M[Rows * M_Cols + k] * N[Cols + k * N_Cols];
                                        }
                                        P[Cols + Rows * N_Cols] = sum;
                                    }
                               }



/**
 * @brief TILE matrix multiplication P=MN.
 * @param P output matrix size M_RowsxN_Cols.
 * @param M input matrix size M_RowsxM_Cols.
 * @param N input matrix size M_ColsxN_Cols. (M_Cols=N_Rows)
 * @note implementation without thread coarsing. 
 */
__global__
void TileMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int M_Rows, const int M_Cols, const int N_Cols)
                               {
                                    __shared__ float Mds[TILE_SIZE][TILE_SIZE];
                                    __shared__ float Nds[TILE_SIZE][TILE_SIZE];
                                    int tx = threadIdx.x;   int ty = threadIdx.y;
                                    int bx = blockIdx.x;    int by = blockIdx.y;
                                    
                                    int Row = ty + TILE_SIZE * by;
                                    int Col = tx + TILE_SIZE * bx;

                                    int numTiles = (M_Cols - 1 + TILE_SIZE) / TILE_SIZE;
                                    float sum = 0.0f;

                                    for (int ph=0; ph < numTiles; ph++)
                                    {
                                        // Load data to shared memory
                                        Mds[ty][tx] = ((Row < M_Rows) && (tx + ph * TILE_SIZE) < M_Cols) ? M[Row * M_Cols + tx + ph * TILE_SIZE] : 0.0f;
                                        Nds[ty][tx] = ((ty + ph * TILE_SIZE < M_Cols) && (Col < N_Cols)) ? N[Col + N_Cols * (ty + ph * TILE_SIZE)] : 0.0f;
                                        __syncthreads();

                                        for (int k=0; k < TILE_SIZE; k++)
                                        {
                                            sum += Mds[ty][k] * Nds[k][tx];
                                        }
                                        __syncthreads();
                                    }
                                    
                                    if ((Row < M_Rows) && (Col < N_Cols))
                                    {
                                        P[Col + Row * N_Cols] = sum;
                                    }
                               }


/**
* @brief TILE matrix multiplication P=MN.
* @param P output matrix size rmxcn.
* @param M input matrix size rmxcm.
* @param N input matrix size cmxcn. (cm=cr)
* @note implementation with tile and thread coarsing not optimize occupancy yet. 
*/
__global__
void CoarseMatrixMultiplication(float* __restrict__ P,
                               const float* __restrict__ M,
                               const float* __restrict__ N,
                               const int rm, const int cm, const int cn)
                               {
                                     // One M tile is reused for all COARSE_FACTOR output columns,
                                     // but each output column needs its OWN N tile -> extra coarse dim.
                                     __shared__ float Mds[TILE_SIZE][TILE_SIZE];
                                     __shared__ float Nds[TILE_SIZE][TILE_SIZE * COARSE_FACTOR];
                                     int tx = threadIdx.x;   int ty = threadIdx.y;
                                     int bx = blockIdx.x;    int by = blockIdx.y;

                                     int Row = ty + by * TILE_SIZE;
                                     int Colstart = tx + bx * TILE_SIZE * COARSE_FACTOR;

                                     int numTiles = (cm - 1 + TILE_SIZE) / TILE_SIZE;
                                     float sum[COARSE_FACTOR] = {0.0f};

                                     for (int ph=0; ph < numTiles; ph++)
                                     {
                                         // load data to shared memory
                                         int Mcol = tx + ph * TILE_SIZE;
                                         int Nrow = ty + ph * TILE_SIZE;
                                         Mds[ty][tx] = ((Row < rm) && (Mcol < cm)) ? M[Row*cm + Mcol] : 0.0f;

                                         // FIX: load COARSE_FACTOR distinct N tiles into separate columns
                                         for (int c=0; c < COARSE_FACTOR; c++)
                                         {
                                             int col = Colstart + c * TILE_SIZE;
                                             Nds[ty][tx + c * TILE_SIZE] =
                                                 ((Nrow < cm) && (col < cn)) ? N[col + Nrow*cn] : 0.0f;
                                         }
                                         __syncthreads();

                                         // Compute: reuse Mds row, multiply against each N tile column block
                                         for (int k=0; k < TILE_SIZE; k++)
                                         {
                                             float m = Mds[ty][k];              // FIX: load M element once, reuse
                                             for (int c=0; c < COARSE_FACTOR; c++)
                                             {
                                                 sum[c] += m * Nds[k][tx + c * TILE_SIZE];  // FIX: index c-th N tile
                                             }
                                         }
                                         __syncthreads();   // FIX: single end-of-phase barrier
                                     }

                                     for (int c=0; c < COARSE_FACTOR; c++)
                                     {
                                         int col = Colstart + c*TILE_SIZE;
                                         if ((Row < rm) && (col < cn))          // FIX: guard the store
                                         {
                                             P[Row * cn + col] = sum[c];
                                         }
                                     }
                               }