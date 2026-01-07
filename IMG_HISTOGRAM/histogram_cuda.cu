#include "histogram_cuda.cuh"

/**
 * @brief Implement histogram for BGR image
 * @param data - input image layout as row major vector
 * @param num_pixels - width * height
 * @param histo - histogram results
 */
__global__
void bgr_hist_aggregate_kernel(const unsigned char* __restrict__ data,
                                unsigned int num_pixels,
                                unsigned int* __restrict__ histo)
{
    // 3 separate shared memory histograms
    __shared__ unsigned int h_smem_b[NUM_BINS];
    __shared__ unsigned int h_smem_g[NUM_BINS];
    __shared__ unsigned int h_smem_r[NUM_BINS];
    
    // Initialize all 3
    for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x) {
        h_smem_b[bin] = 0u;
        h_smem_g[bin] = 0u;
        h_smem_r[bin] = 0u;
    }
    __syncthreads();

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int step = blockDim.x * gridDim.x;

    // Process BGR pixels (3 bytes per pixel)
    for (unsigned int i=tid; i < num_pixels; i+=step) {
        unsigned int pixel_idx = i * 3;
        unsigned char b = data[pixel_idx + 0];
        unsigned char g = data[pixel_idx + 1];
        unsigned char r = data[pixel_idx + 2];
        
        unsigned int bin_b = b >> BIN_SHIFT;
        unsigned int bin_g = g >> BIN_SHIFT;
        unsigned int bin_r = r >> BIN_SHIFT;
        
        atomicAdd(&h_smem_b[bin_b], 1u);
        atomicAdd(&h_smem_g[bin_g], 1u);
        atomicAdd(&h_smem_r[bin_r], 1u);
    }
    __syncthreads();

    // Merge to global memory (3 separate regions)
    for (unsigned int bin=threadIdx.x; bin < NUM_BINS; bin+=blockDim.x) {
        if (h_smem_b[bin] > 0)
            atomicAdd(&histo[bin], h_smem_b[bin]);
        if (h_smem_g[bin] > 0)
            atomicAdd(&histo[NUM_BINS + bin], h_smem_g[bin]);
        if (h_smem_r[bin] > 0)
            atomicAdd(&histo[2*NUM_BINS + bin], h_smem_r[bin]);
    }
}

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
                                                atomicAdd(&h_smem[prevBin], acc);
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


/**
 * @brief Main histogram implementation
 * @param img gray image
 * @param h_hist host histogram output pointer
 * @note we assume that kernel work on Gray image level
 * @note the kernel release the GPU memory after calculation but when finsh we need relese
 * the host memory
 */
__host__
void gray_histogram_main(cv::Mat img, unsigned int** h_hist)
{
    if (img.type() != CV_8UC1)
    {
        printf("Error in gray_histogram_main: type diffrent from CV_8UC1\n");
        return;
    }

    // Allocate host memory
    unsigned int NUM_PIXELS = static_cast<unsigned int>(img.cols * img.rows);
    size_t SizeInBytes_hist = sizeof(unsigned int) * NUM_BINS;
    size_t SizeInBytes_img = sizeof(unsigned char) * NUM_PIXELS;
    cudaMallocHost((void**)h_hist, SizeInBytes_hist);

    // Allocate device Memory
    unsigned char* d_img;
    unsigned int* d_hist;

    cudaMalloc((void**)&d_img, SizeInBytes_img);
    cudaMalloc((void**)&d_hist, SizeInBytes_hist);

    // Initialize histogram to zero
    cudaMemset(d_hist, 0, SizeInBytes_hist);

    // Copy data from host to device
    cudaMemcpy(d_img, img.data, SizeInBytes_img, cudaMemcpyHostToDevice);

    // Define Grid Size
    dim3 NumThreadPerBlock(256, 1, 1);
    dim3 GridSize;
    GridSize.z = 1;
    GridSize.y = 1;
    GridSize.x = (NUM_PIXELS + NumThreadPerBlock.x - 1) / NumThreadPerBlock.x;
    
    // Launch Kernel
    gray_hist_aggregate_kernel <<<GridSize, NumThreadPerBlock>>> (d_img, NUM_PIXELS, d_hist);
    cudaDeviceSynchronize();

    // Move data from device to host
    cudaMemcpy(*h_hist, d_hist, SizeInBytes_hist, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_hist);
    cudaFree(d_img);

    return;
}