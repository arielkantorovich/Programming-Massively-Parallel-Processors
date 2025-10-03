#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CHANNELS 3

/**
 * @brief The following kernel method convert RGB image to gray.
 * @param Pin - input image HxWxCHANNEl where CHANNEL (3)
 * @param Pout - gray output image CHANNEL (1)
 * @note  Because we read opencv matrix that is BGR format
 * the channel offset is swep between red and blue.
 * 
*/
__global__
void colorToGreyscaleConversion(unsigned char *Pout, 
                                unsigned char *Pin, 
                                int width, int height)
{
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    if (Row < height && Col < width)
    {
        int gridOffset = Col + Row * width;
        int channelOffset = CHANNELS * gridOffset;
        unsigned char r = Pin[channelOffset + 2];
        unsigned char g = Pin[channelOffset + 1];
        unsigned char b = Pin[channelOffset + 0];
        Pout[gridOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


int main(int argc, char **argv)
{
    // Define path's
    const char* input_path = "Images/lenna.jpg";
    const char* output_path = "Images/lenna_gray.jpg";

    // read Image
    cv::Mat input_bgr = cv::imread(input_path, cv::IMREAD_COLOR);
    if (input_bgr.empty())
    {
        printf("Error image is empty. Not found path %s \n", input_path);
        return 1;
    }

    int width = input_bgr.cols;
    int height = input_bgr.rows;

    // Allocate Device Memory
    unsigned char *d_in, *d_out;
    size_t inputSizeBytes = width * height * CHANNELS * sizeof(unsigned char);
    size_t outputSizeBytes = width * height * sizeof(unsigned char);
    cudaMalloc((void**) &d_in, inputSizeBytes);
    cudaMalloc((void**) &d_out, outputSizeBytes);

    // Allocate host memory
    cv::Mat result(height, width, CV_8UC1);

    // Transfer data from host to device
    cudaMemcpy(d_in, input_bgr.data, inputSizeBytes, cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 BlockDim(16, 16, 1);
    dim3 GridDim;
    GridDim.x = (width+BlockDim.x-1)/BlockDim.x;
    GridDim.y = (height+BlockDim.y-1)/BlockDim.y;
    GridDim.z = 1;
    colorToGreyscaleConversion <<<GridDim, BlockDim>>>(d_out, d_in, width, height);

    // Wait until device finsh launching kernel caclulation
    cudaDeviceSynchronize();

    // Transfer results to host
    cudaMemcpy(result.data, d_out, outputSizeBytes, cudaMemcpyDeviceToHost);

    // Save results
    cv::imwrite(output_path, result);

    // Reallocate Memory Device and cou
    result.release();
    input_bgr.release();
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}