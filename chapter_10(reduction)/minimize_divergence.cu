#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstring>

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

__global__ void sum_reduction_reorder_kernel(float* input, float* output)
{
    int tid = threadIdx.x;
    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        if (tid < stride) {
            input[tid] += input[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) *output = input[0];
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./a.out N\n");
        return -1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) return -1;

    float* h_arr = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) h_arr[i] = (float)(i + 1);

    float* h_padd = nullptr;
    int Np = 0;

    cudaError_t e = pad_to_pow2_pinned(h_arr, N, &h_padd, &Np);
    if (e != cudaSuccess) {
        printf("pad error: %s\n", cudaGetErrorString(e));
        return -1;
    }

    if (Np > 2048) {
        printf("Np=%d too large for single-block Np/2 threads.\n", Np);
        return -1;
    }

    float* d_arr = nullptr;
    float* d_out = nullptr;
    cudaMalloc((void**)&d_arr, sizeof(float) * Np);
    cudaMalloc((void**)&d_out, sizeof(float));

    cudaMemcpy(d_arr, h_padd, sizeof(float) * Np, cudaMemcpyHostToDevice);

    dim3 threadsInBlock(Np / 2, 1, 1);
    sum_reduction_reorder_kernel<<<1, threadsInBlock>>>(d_arr, d_out);

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) printf("Kernel launch error: %s\n", cudaGetErrorString(ke));
    cudaDeviceSynchronize();

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum=%f\n", h_out);

    free(h_arr);
    cudaFreeHost(h_padd);
    cudaFree(d_arr);
    cudaFree(d_out);
    return 0;
}
