#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * @brief Auxilary method the print results on host
*/
void printResults(unsigned int* hist)
{
    const char *x[7] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int i=0; i < 7; i++)
    {
        printf("%s = %i \n", x[i], hist[i]);
    }
}

/**
 * @brief Histogram GPU implementation using atomic operation
 * @param data - input
 * @param length - input length
 * @param hist - 1D histogram output
 * @note input is sequense of characters (1-D)
*/
__global__ 
void hist_kernel(char *data, unsigned int length, unsigned int *hist)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >=0 && alphabet_position < 26)
        {
            atomicAdd(&(hist[alphabet_position/4]), 1);
        }
    }
}




int main(int argc, char **argv)
{
    // Some constant definition
    const char *seq = "programming massively parallel processors";
    unsigned int N = 42;

    // allocate host mem
    unsigned int h_hist[N] = {0};
    
    // allocate device Memory
    unsigned int *d_hist;
    char *d_in;
    size_t SizeInBytesHist = sizeof(unsigned int) * N;
    size_t SizeInBytesInput = sizeof(char) * N;
    cudaMalloc(&d_hist, SizeInBytesHist);
    cudaMalloc(&d_in, SizeInBytesInput);

    // Transfer data from host to device
    cudaMemcpy(d_hist, &h_hist, SizeInBytesHist, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, seq, SizeInBytesInput, cudaMemcpyHostToDevice);

    // Define cuda grid size
    dim3 ThreadsInblock(64, 1, 1);
    dim3 GridSize((N + ThreadsInblock.x - 1) / ThreadsInblock.x, 1, 1);

    // Launch cuda Kernel
    hist_kernel <<<GridSize, ThreadsInblock>>> (d_in, N, d_hist);
    cudaDeviceSynchronize();

    // Copy data from device back to host
    cudaMemcpy(h_hist, d_hist, SizeInBytesHist, cudaMemcpyDeviceToHost);

    printResults(h_hist);

    // deallocate memory
    cudaFree(d_hist);
    cudaFree(d_in);

    return 0;
}