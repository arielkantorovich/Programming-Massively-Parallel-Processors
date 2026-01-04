#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_BINS 7
#define NUM_CHAR_PER_BIN 4

/**
 * @brief Histogram aggragtion each thread 
 * aggregate consecutive updates into a single update
 * @param data Input data array.
 * @param length Length of the input data array.
 * @param histo Output histogram array.
 */
__global__
void aggragte_histogram_kernel(const char* __restrict__ data,
                            unsigned int length,
                            unsigned int* __restrict__ histo)
                            {
                                // Initialize shared Memory
                                __shared__ unsigned int h_s[NUM_BINS];

                                for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
                                {
                                    h_s[bin] = 0u;
                                }
                                __syncthreads();

                                // Load data to shared mem and define acc
                                unsigned int acc = 0u;
                                int prevBin = -1;

                                unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                unsigned int step = blockDim.x*gridDim.x;
                                for (unsigned int i=tid; tid < length; tid+=step)
                                {
                                    int alphabeta_position = data[i] - 'a';
                                    if (alphabeta_position >= 0 && alphabeta_position < 26)
                                    {
                                        unsigned int bin = alphabeta_position / NUM_CHAR_PER_BIN;
                                        if (bin == prevBin)
                                        {
                                            acc ++;
                                        }
                                        else
                                        {
                                            if (acc > 0)
                                            {
                                                atomicAdd(&h_s[prevBin], acc);
                                            }
                                            prevBin = bin;
                                            acc = 1;
                                        }
                                    }
                                }
                                
                                // Before continue load the last acc
                                if (acc > 0)
                                {
                                    atomicAdd(&h_s[prevBin], acc);
                                }
                                __syncthreads();

                                // Merge to private copys
                                for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
                                {
                                    unsigned int value = h_s[bin];
                                    if (value > 0)
                                    {
                                        atomicAdd(&histo[bin], value);
                                    }
                                }
                                
                            }