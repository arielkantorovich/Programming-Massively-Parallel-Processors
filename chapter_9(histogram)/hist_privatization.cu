/**
 * @author Ariel Kantorovich
 * @brief Implement Histogram using privatization concept The idea is to replicate highly contended output data structures 
 * into private copies so that each subset of threads can update its private copy.
 * Then merge all the copies to final results.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define NUM_BINS 7
#define NUM_CHAR_PRE_BIN 4

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

/**
 * @brief Calculate histogram using private copy technique
 * @param data - string of characters
 * @param N - length of data
 * @param hist - Histogram which contain all the copies mean NUM_BINS * GridDim Size 
*/
__global__
void histo_private_kernel(const char* __restrict__ data,
                          unsigned int N,
                          unsigned int* __restrict__ histo)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    // No early return (important!)
    if (i < N) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            unsigned int bin = (unsigned int)(alphabet_position / NUM_CHAR_PRE_BIN);
            atomicAdd(&histo[blockIdx.x * NUM_BINS + bin], 1);
        }
    }

    // Make sure the block finished updating its private bins
    __syncthreads();

    // Merge: blocks 1..gridDim-1 add their private bins into histo[0..NUM_BINS-1]
    if (blockIdx.x > 0) {
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int value = histo[blockIdx.x * NUM_BINS + bin];
            if (value) atomicAdd(&histo[bin], value);
        }
    }
}


int main()
{
    const char* seq = "programming massively parallel processors";
    unsigned int N = (unsigned int)std::strlen(seq);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    // Allocate device input
    char* d_in = nullptr;
    checkCuda(cudaMalloc((void**)&d_in, N * sizeof(char)), "cudaMalloc d_in");
    checkCuda(cudaMemcpy(d_in, seq, N * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy input");

    // Allocate histogram: NUM_BINS * blocks (private per block)
    unsigned int* d_histo = nullptr;
    checkCuda(cudaMalloc((void**)&d_histo, blocks * NUM_BINS * sizeof(unsigned int)), "cudaMalloc d_histo");
    checkCuda(cudaMemset(d_histo, 0, blocks * NUM_BINS * sizeof(unsigned int)), "cudaMemset d_histo");

    // Launch
    histo_private_kernel<<<blocks, threads>>>(d_in, N, d_histo);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    // Copy back only the final bins (first NUM_BINS entries)
    unsigned int h_histo[NUM_BINS] = {0};
    checkCuda(cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost),
              "cudaMemcpy output");

    const char* labels[NUM_BINS] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int b = 0; b < NUM_BINS; ++b) {
        printf("%s = %u\n", labels[b], h_histo[b]);
    }

    cudaFree(d_in);
    cudaFree(d_histo);
    return 0;
}