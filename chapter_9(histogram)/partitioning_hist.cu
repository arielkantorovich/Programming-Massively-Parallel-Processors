#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 7
#define NUM_CHAR_PRE_BIN 4
#define C_FACTOR 6

/**
 * @brief Kernel to compute histogram using partitioning method.
 * @param data Input data array.
 * @param data_len Length of the input data array.
 * @param histo Output histogram array.
 * @note we assume that have block keep some partial compy. of histogram in shared memory.
 */

 __global__
void partitioning_hist_kernel(const char* __restrict__ data, 
                              unsigned int data_len,
                              unsigned int* __restrict__ histo)
{
    __shared__ unsigned int h_s[NUM_BINS];
    // Initalize histogram
    for (int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
    {
        h_s[bin] = 0u;
    }
    __syncthreads();

    // Load using partitioning teachnique
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned int i=tid*C_FACTOR; i < min((tid+1) * C_FACTOR, data_len); i++)
    {
        int alphabet_position = data[i] - 'a';
        if ((alphabet_position >= 0) && (alphabet_position < 26))
        {
            atomicAdd(&h_s[alphabet_position/NUM_CHAR_PRE_BIN], 1);
        }
    }
    __syncthreads();

    // Merge to results on the blockIdx 0
    for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
    {
        int value = h_s[bin];
        if (value > 0)
        {
            atomicAdd(&histo[bin], value);
        }
    }
}
