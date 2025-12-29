/**
 * @author Ariel Kantorovich
 * @brief benefit of creating a private copy of the histogram on a per-thread-block 
 * basis is that if the number of bins in the histogram is small enough, 
 * the private copy of the histogram can be declared in shared memory. 
 * Using shared memory would not be possible if the private copy were accessed by 
 * multiple blocks because blocks do not have visibility of each other’s shared memory.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 7
#define NUM_CHAR_PRE_BIN 4

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}


#define NUM_BINS 7
#define NUM_CHAR_PER_BIN 4

__global__
void hist_smem(const char* __restrict__ data, unsigned int N, unsigned int* histo)
{
    __shared__ unsigned int h_s[NUM_BINS];

    // init shared histogram
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        h_s[bin] = 0u;

    __syncthreads();

    // update shared histogram
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&h_s[alphabet_position / NUM_CHAR_PER_BIN], 1);
        }
    }

    __syncthreads();

    // merge shared histogram into global histogram
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int value = h_s[bin];
        if (value) atomicAdd(&histo[bin], value);
    }
}


int main()
{
    const char* seq = "programming massively parallel processors";
    unsigned int N = (unsigned int)strlen(seq);

    // device input
    char* d_in = nullptr;
    checkCuda(cudaMalloc((void**)&d_in, N * sizeof(char)), "cudaMalloc d_in");
    checkCuda(cudaMemcpy(d_in, seq, N * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy input");

    // device histogram (only NUM_BINS!)
    unsigned int* d_histo = nullptr;
    checkCuda(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)), "cudaMalloc d_histo");
    checkCuda(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)), "cudaMemset d_histo");

    // launch
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    hist_smem<<<blocks, threads>>>(d_in, N, d_histo);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    // copy result back
    unsigned int h_histo[NUM_BINS] = {0};
    checkCuda(cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost), "cudaMemcpy output");

    // print
    const char* labels[NUM_BINS] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int b = 0; b < NUM_BINS; ++b) {
        printf("%s = %u\n", labels[b], h_histo[b]);
    }

    cudaFree(d_in);
    cudaFree(d_histo);
    return 0;
}