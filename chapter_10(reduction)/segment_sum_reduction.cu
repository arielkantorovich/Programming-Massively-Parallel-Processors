#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define BLOCK_DIM 256

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
 * @brief sub reduction array to arbitrary length using segment technique
 * @param input array with length of 2^k if not do padding before
 * @param output scalar the return the sum of the array
 * @note we assume blockDim = 0.5 * segment array
 */
__global__ void segment_sum_reduction(const float* __restrict__ input,
                                      float* output)
{
    __shared__ float smem[BLOCK_DIM];
    unsigned int tid = threadIdx.x;
    unsigned int start = 2 * blockDim.x * blockIdx.x;

    // Load 2 elements per thread (assumes padding / valid range)
    smem[tid] = input[start + tid] + input[start + blockDim.x + tid];
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // One atomic per block
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Error: please run ./<exe> <N> <num> \n where <N> is the array length. \n <num> is the constant of the array \n.");
    }

    int n = atoi(argv[1]);
    int num = atoi(argv[2]);
    if (n <= 0) return -1;

    // Allocate memory in host
    float *h_in, *h_out;
    h_in = (float*)malloc(sizeof(float) * n);
    h_out = (float*)malloc(sizeof(float));
    
    float *h_in_padded = nullptr;
    int Np = 0;

    for (int i=0; i < n; i++)
    {
        h_in[i] = num;
    }

    cudaError_t e = pad_to_pow2_pinned(h_in, n, &h_in_padded, &Np);
    if (e != cudaSuccess)
    {
        printf("pad error: %s\n", cudaGetErrorString(e));
        return -1;
    }

    // Allocate device memory
    float *d_pad, *d_output;
    cudaMalloc((void**)&d_pad, sizeof(float)*Np);
    cudaMalloc((void**)&d_output, sizeof(float));
    cudaMemset(d_output, 0, sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_pad, h_in_padded, sizeof(float)*Np, cudaMemcpyHostToDevice);

    // Define and launch grid
    dim3 threadsInBlock(BLOCK_DIM, 1, 1);
    dim3 NumBlocks((Np + 2*BLOCK_DIM - 1) / (2*BLOCK_DIM), 1, 1);
    segment_sum_reduction <<< NumBlocks, threadsInBlock >>> (d_pad, d_output);
    cudaDeviceSynchronize();

    // Transfer results
    cudaMemcpy(h_out, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output = %f \n", *h_out);

    // Deallocate Memory
    cudaFree(d_pad);  cudaFree(d_output);
    free(h_in); free(h_out); cudaFreeHost(h_in_padded);
    return 0;
}