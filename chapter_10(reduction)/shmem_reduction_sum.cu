#include <stdio.h>
#include <stdlib.h>
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

    std::memset(*h_padded, 0, sizeof(float) * Np);
    std::memcpy(*h_padded, h_in, sizeof(float) * N);
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
    __shared__ float input_s[blockDim.x];
    unsigned int tid = threadIdx.x;
    // Load sum from global to shared memory iteration-1
    input_s = input[tid] + input[tid+blockDim.x];
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