#include "histogram_cuda.cuh"


/**
 * @brief Implement histogram for gray image (channel = 1)
 * @param data - input image layout as row major vector
 * @param num_pixels - width * height
 * @param histo - histogram results
 */
__global__
void gray_hist_aggregate_kernel(const unsigned char* __restrict__ data,
                                unsigned int num_pixels,
                                unsigned int* __restrict__ histo)
                                {
                                    // Initialize shared Memory
                                    __shared__ unsigned int h_smem[NUM_BINS];
                                    for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
                                    {
                                        h_smem[bin] = 0u;
                                    }
                                    __syncthreads();

                                    // Upload data to private histogram (per block)
                                    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                    unsigned int step = blockDim.x * gridDim.x;

                                    unsigned int acc = 0u;
                                    int prevBin = -1;

                                    for (unsigned int i=tid; i < num_pixels; i+=step)
                                    {
                                        unsigned int pixel = (unsigned int)data[i];
                                        unsigned int bin = pixel >> BIN_SHIFT;
                                        if ((int)bin == prevBin)
                                        {
                                            acc ++;
                                        }
                                        else
                                        {
                                            if (acc > 0)
                                            {
                                                atomicAdd(&h_smem[i], acc);
                                            }
                                            
                                            prevBin = (int)bin;
                                            acc = 1u;
                                        }
                                    }
                                    
                                    // load the final bin
                                    if (acc > 0 && prevBin >= 0)
                                    {
                                        atomicAdd(&h_smem[prevBin], acc);
                                    }
                                    __syncthreads();

                                    // Merge the final results
                                    for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x)
                                    {
                                        unsigned int value = h_smem[bin];
                                        if (value > 0)
                                        {
                                            atomicAdd(&histo[bin], h_smem[bin]);
                                        }
                                    }
                                }