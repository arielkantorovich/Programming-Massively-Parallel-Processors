#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/**
 * @brief Element weise matrix addition.
 * Implement threed Book q1 option A and B.
 * @param A - input matrix
 * @param B - input matrix
 * @param C - output element wise matrix C[i][j]  = A[i][j] + B[i][j]
*/
__global__
void addMatrices_B(float *A, float *B, float *C, int n)
{
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    if (Row >=n || Col>=n)
        return;
    C[Col + Row * n] = A[Col + Row * n] + B[Col + Row * n];
}


/**
 * @brief Element weise matrix addition.
 * Implement threed Book q1 option A and B.
 * @param A - input matrix
 * @param B - input matrix
 * @param C - output element wise matrix C[i][j]  = A[i][j] + B[i][j]
 * @note each thread to produce one output matrix row
*/
__global__
void addMatrices_C(float *A, float *B, float *C, int n)
{
    int Row = threadIdx.y + blockIdx.y*blockDim.y;
    if (Row >= n)
        return;
    for (int Col=0; Col < n; Col++)
    {
        C[Col + Row*n] = A[Col + Row*n] + B[Col + Row*n];
    } 
}


/**
 * @brief Element weise matrix addition.
 * Implement threed Book q1 option A and B.
 * @param A - input matrix
 * @param B - input matrix
 * @param C - output element wise matrix C[i][j]  = A[i][j] + B[i][j]
 * @note each thread to produce one output matrix col
*/
__global__
void addMatrices_D(float *A, float *B, float *C, int n)
{
    int Col = threadIdx.x + blockDim.x*blockIdx.x;
    if (Col >=n)
        return;
    for (int Row=0; Row < n; Row++)
    {
        C[Col + Row*n] = A[Col + Row*n] + B[Col + Row*n];
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Error need run ./exe <n> <Option> \n * Where <n> is matrix size \n * Where <Option> can be B, C, or D \n");
        return 1;
    }
    
    // Define global Parameters
    int n = atoi(argv[1]);
    char *Option = argv[2];
    size_t SizeInBytes = n * n * sizeof(float);

    // Allocate Host matrixes Memory
    float *h_a, *h_b, *h_c;
    h_a = (float *)malloc(SizeInBytes);
    h_b = (float *)malloc(SizeInBytes);
    h_c = (float *)malloc(SizeInBytes);

    for (int i=0; i < n; i++)
    {
        for (int j=0; j < n; j++)
        {
            h_a[j + i*n] = 1.70f;
            h_b[j + i*n] = 2.70f;
        }
    }

    // Allocate Device Memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, SizeInBytes);
    cudaMalloc((void **)&d_b, SizeInBytes);
    cudaMalloc((void **)&d_c, SizeInBytes);

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch Kernel
    if (*Option == 'B')
    {
        dim3 BlockDim(16, 16, 1);
        dim3 GridDim;
        GridDim.x = (n + BlockDim.x - 1) / BlockDim.x;
        GridDim.y = (n + BlockDim.y - 1) / BlockDim.y;

        addMatrices_B<<<GridDim, BlockDim>>>(d_a, d_b, d_c, n);
    }

    else if (*Option == 'C')
    {
        dim3 BlockDim(1, 256, 1);
        dim3 GridDim;
        GridDim.y = (n + BlockDim.y - 1) / BlockDim.y;
        addMatrices_C<<<GridDim, BlockDim>>>(d_a, d_b, d_c, n);
    }
    // 'D'
    else
    {
        dim3 BlockDim(256, 1, 1);
        dim3 GridDim;
        GridDim.x = (n + BlockDim.x - 1) / BlockDim.x;
        addMatrices_D<<<GridDim, BlockDim>>>(d_a, d_b, d_c, n);
    }

    // Wait until Launch kernel will finsh
    cudaDeviceSynchronize();

    // Transfer results to Host
    cudaMemcpy(h_c, d_c, SizeInBytes, cudaMemcpyDeviceToHost);

    // Print results
    for (int i=0; i < n; i++)
    {
        for (int j=0; j < n; j++)
        {
            printf("C[%i][%i] = %f \n", i, j, h_c[j+n*i]);
        }
    }

    // Reallocate all memory Device and Host
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}