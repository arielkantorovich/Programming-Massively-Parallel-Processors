#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cuda_runtime.h>

/**
 * @param h_in input array
 * @param N - original length
 * @param h_padded new padded array in size 2^K
 * @param Np_out the new length
 */
cudaError_t pad_to_pow2_pinned(const float* h_in, int N,
                               float** h_padded, int* Np_out)
{
    if (!h_in || !h_padded || !Np_out || N <= 0)
        return cudaErrorInvalidValue;

    int Np = 1;
    while (Np < N) Np <<= 1;
    *Np_out = Np;

    cudaError_t e = cudaMallocHost((void**)h_padded, sizeof(float) * Np);
    if (e != cudaSuccess) return e;

    memset(*h_padded, 0, sizeof(float) * Np);
    memcpy(*h_padded, h_in, sizeof(float) * N);
    return cudaSuccess;
}


/**
 * @brief sum reduction on one block given 1D array return the sum of elements
 * We use shared mem and reorder to optimize the process
 * @param input 1D array
 * @param output sum results scaler float
 * @note N = 2^K and N < 2048 , blockDim = N/2
 */
__global__
void sum_shmem_reduction_kernel(float* __restrict__ input, float* __restrict__ output)
{
    extern __shared__ float input_s[];
    unsigned int tid = threadIdx.x;
    // Load sum from global to shared memory iteration-1
    input_s[tid] = input[tid] + input[tid+blockDim.x];
    for (unsigned int stride=blockDim.x/2; stride >=1; stride /=2)
    {
        __syncthreads();
        if (tid < stride)
        {
            input_s[tid] += input_s[tid + stride];
        }
    }

    if (tid == 0)
    {
        *output = input_s[0];
    }
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Error: usage ./<exe> <N>\nWhere <N> is the array size (will be padded to power of 2)\n");
        return -1;
    }

    int N = atoi(argv[1]);
    if (N <= 0 || N > 2048)
    {
        printf("Error: N must be between 1 and 2048\n");
        return -1;
    }

    // Allocate and initialize host memory
    float *h_arr = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++)
    {
        h_arr[i] = i + 1;
    }

    // Pad to power of 2
    float *h_padded;
    int Np;
    cudaError_t err = pad_to_pow2_pinned(h_arr, N, &h_padded, &Np);
    if (err != cudaSuccess)
    {
        printf("Error padding array: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return -1;
    }

    // Allocate device memory
    float *d_arr, *d_out;
    cudaMalloc((void**)&d_arr, sizeof(float) * Np);
    cudaMalloc((void**)&d_out, sizeof(float));

    // Transfer to device
    cudaMemcpy(d_arr, h_padded, sizeof(float) * Np, cudaMemcpyHostToDevice);

    // Launch kernel with Np/2 threads and shared memory
    dim3 threadsInBlock(Np / 2, 1, 1);
    dim3 GridSize(1, 1, 1);
    size_t sharedMemSize = sizeof(float) * (Np / 2);
    sum_shmem_reduction_kernel<<<GridSize, threadsInBlock, sharedMemSize>>>(d_arr, d_out);
    cudaDeviceSynchronize();

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    // Transfer result back
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum=%f\n", h_out);

    // Cleanup
    free(h_arr);
    cudaFreeHost(h_padded);
    cudaFree(d_arr);
    cudaFree(d_out);

    return 0;
}