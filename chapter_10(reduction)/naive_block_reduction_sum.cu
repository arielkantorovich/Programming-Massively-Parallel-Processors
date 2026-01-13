#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


/**
 * @brief sum reduction on one block given 1D array return the sum of elements
 * @param input 1D array
 * @param output sum results scaler float
 * @note Thie method assume the all the array is inside one block mean N <=2048
 */
__global__
void sum_reduction_block(float *input, float *output)
{
    int i = threadIdx.x * 2;
    for (int stride=1; stride < blockDim.x; stride*=2)
    {
        if ((threadIdx.x % stride) == 0)
        {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        *output = input[0];
    }
}



int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Error argc >2: you need run ./<exe> <N> \n Where <N> is the array size need to be smaller from 2048. \n");
        return -1;
    }

    int N = atoi(argv[1]);
    if (N > 2048)
    {
        printf("Error: The naive algorithm assume that all elemnts in the array exsist in one block which mean N < 2048 \n");
        return -1;
    }

    // Allocate host memory
    float *h_arr, *h_out;
    size_t SizeInBytes = sizeof(float) * N;
    cudaMallocHost((void**)&h_arr, SizeInBytes);
    h_out = (float*)malloc(sizeof(float));
    for (int i=0; i < N; i++)
    {
        h_arr[i] = i+1;
    }
    
    // Allocate Device Memory
    float *d_arr, *d_out;
    cudaMalloc((void**)&d_arr, SizeInBytes);
    cudaMalloc((void**)&d_out, sizeof(float));

    // Transfer memory from host to device
    cudaMemcpy(d_arr, h_arr, SizeInBytes, cudaMemcpyHostToDevice);

    // Define GridSize and launch kernel
    dim3 threadsInBlock(N, 1, 1);
    dim3 GridSize(1, 1, 1);
    sum_reduction_block <<<GridSize, threadsInBlock>>>(d_arr, d_out);
    cudaDeviceSynchronize();

    // Transfer data from device back to host
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum=%f \n", *h_out);

    // deallocate memory
    free(h_out);
    cudaFreeHost(h_arr);
    cudaFree(d_arr);    cudaFree(d_out);

}