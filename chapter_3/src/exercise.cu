#include "exercise.cuh"

/**
 * @brief Naive matrix multiplication implementation.
 * @param M input matrix size n x m.
 * @param N input matrix size m x l.
 * @param P output matrix size n x l.
 */
__global__
void MatrixMut_kernel(float *M, float *N, float *P, int n, int m, int l)
{
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    if (Row < n && Col < l)
    {
        float Pvalue = 0;
        for (int k = 0; k < m; ++k)
        {
            Pvalue += M[Row * m + k] * N[k * l + Col];
        }
        P[Row * l + Col] = Pvalue;
    }
}



/**
 * @brief Each thread computes one entire output ROW of P.
 */
__global__
void MatrixMut_1a(float *M, float *N, float *P, int n, int m, int l)
{
    int Row = threadIdx.x + blockDim.x * blockIdx.x;
    if (Row < n)
    {
        for (int j = 0; j < l; j++)  
        {
            float Pval = 0;
            for (int k = 0; k < m; k++)
            {
                Pval += M[Row * m + k] * N[k * l + j];
            }
            P[Row * l + j] = Pval;     
        }
    }
}

/**
 * @brief matrix multiplication implementation each thread calculate the entire col.
 * @param M input matrix size n x m.
 * @param N input matrix size m x l.
 * @param P output matrix size n x l.
 */
__global__
void MatrixMut_1b(float *M, float *N, float *P, int n, int m, int l)
{
    int Col = threadIdx.x + blockDim.x * blockIdx.x;
    if (Col < l)
    {
        for (int i=0; i < n; ++i)
        {
            float Pval = 0;
            for (int k=0; k < m; ++k)
            {
                Pval += M[i * m + k] * N[k * l + Col];
            }
            P[Col + i*l] = Pval;
        }
    }
}




/**
 * @brief Vector Matrix multiplication
 * @param h_A - output vector size nx1.
 * @param h_B - input matrix size nxn.
 * @param h_C - input vector size nx1.
 * @param n - input size.
 */
void MatVecMult_device(float *h_A, float *h_B, float *h_C, int n)
{
    // allocate device memory
    float *d_A, *d_B, *d_C;
    size_t SizeVecBytes = sizeof(float) * n;
    size_t SizeMatBytes = sizeof(float) * n * n;

    cudaMalloc((void**)&d_A, SizeVecBytes);
    cudaMalloc((void**)&d_B, SizeMatBytes);
    cudaMalloc((void**)&d_C, SizeVecBytes);
    
    // Transfer data from host to memory
    cudaMemcpy(d_B, h_B, SizeMatBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, SizeVecBytes, cudaMemcpyHostToDevice);

    // Define grid dim
    dim3 ThreadPerBlock(256, 1, 1);
    dim3 GridDim;
    GridDim.x = (n-1 + ThreadPerBlock.x) / ThreadPerBlock.x;

    // launch kernel
    MatVecMult_kernel <<<GridDim, ThreadPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Transfer results to host
    cudaMemcpy(h_A, d_A, SizeVecBytes, cudaMemcpyDeviceToHost);

    // deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * @brief Vector Matrix multiplication kernel.
 * @param A - output vector size nx1.
 * @param B - input matrix size nxn.
 * @param C - input vector size nx1.
 * @param n - input size.
 * @note __restrict__ is a contract you make with the compiler:
 *  "these pointers never alias — the memory regions they point to don't overlap." 
 * So a write through A can't possibly change what B or C point to.
 * Without it, the compiler must assume the worst case: that A, B, C might overlap. That pessimism blocks some optimizations.
 */
__global__
void MatVecMult_kernel(float*  __restrict__ A, const float*  __restrict__ B, const float*  __restrict__ C, int n)
{
    int Row = threadIdx.x + blockIdx.x * blockDim.x;
    if (Row < n)
    {
        float Pvalue = 0;
        for (int k = 0; k < n; ++k)
        {
            Pvalue += B[Row * n + k] * C[k];
        }
        A[Row] = Pvalue;
    }
}
