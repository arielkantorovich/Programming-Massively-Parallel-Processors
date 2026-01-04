#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 7
#define NUM_CHAR_PER_BIN 4

/**
 * @brief  Histogram kernel with coarsening using interleaved partitioning
 * @param data Input data array.
 * @param length Length of the input data array.
 * @param histo Output histogram array.
 */
__global__
void interleaved_hist_kernel(const char* __restrict__ data,
                             unsigned int length,
                             unsigned int* __restrict__ histo)
{
    __shared__ unsigned int h_s[NUM_BINS];

    // init shared histogram
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        h_s[b] = 0u;
    __syncthreads();

    // interleaved partitioning (grid-stride loop)
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int idx = tid; idx < length; idx += stride) {
        int alphabet_position = (int)data[idx] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&h_s[alphabet_position / NUM_CHAR_PER_BIN], 1u);
        }
    }
    __syncthreads();

    // merge shared histogram into global histogram
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        unsigned int v = h_s[b];
        if (v) atomicAdd(&histo[b], v);
    }
}