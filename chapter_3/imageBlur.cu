#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define KERNEL_SIZE 5
#define BLUR_RADIUS ((KERNEL_SIZE) / 2)
#define CHANNEL 3 /*1*/

/**
 * @brief Implement box fiter on device
 * @param Pin input image
 * @param Pout output image
*/
__global__
void ImageBlur(unsigned char *Pin, unsigned char *Pout, int w, int h)
{
    int Col = threadIdx.x + blockDim.x * blockIdx.x;
    int Row = threadIdx.y + blockDim.y * blockIdx.y;

    if (Col < w && Row < h)
    {
        for (int c=0; c < CHANNEL; c++)
        {
            int pixelSum = 0;
            int pixelCount = 0;

            for (int k_row=-BLUR_RADIUS; k_row < BLUR_RADIUS + 1; k_row++)
            {
                for (int k_col=-BLUR_RADIUS; k_col < BLUR_RADIUS + 1; k_col++)
                {
                    int Col_k = Col + k_col;
                    int Row_k = Row + k_row;

                    if ((Col_k >-1 && Col_k < w) && (Row_k > -1 && Row_k < h))
                    {
                        int c_idx = CHANNEL * (Col_k + Row_k*w);
                        pixelSum += Pin[c_idx + c];
                        pixelCount ++;
                    }
                }
            }
            int c_idx = CHANNEL * (Col + Row*w);
            Pout[c_idx+c] = (unsigned char)(pixelSum / pixelCount);
        }
    }
}


int main(int argc, char **argv)
{
    // Read img and allocate host memory
    const char* input_path = "Images/lenna.jpg";
    const char* output_path = "Images/lenna_blur.jpg";
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR/*cv::IMREAD_GRAYSCALE*/);

    if (img.empty())
    {
        printf("Not sucssed to read image path: %s \n", input_path);
        return 1;
    }

    cv::Mat result = cv::Mat(img.rows, img.cols, img.type());

    // Allocate device Memory
    unsigned char *d_in, *d_out;
    size_t SizeInBytes = img.cols * img.rows * img.channels() * sizeof(unsigned char);
    cudaMalloc((void**) &d_in, SizeInBytes);
    cudaMalloc((void**) &d_out, SizeInBytes);

    // Transfer Data from host to device 
    cudaMemcpy(d_in, img.data, SizeInBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 BlockDim(16, 16, 1);
    dim3 GridDim;
    GridDim.x = (img.cols + BlockDim.x - 1) / BlockDim.x;
    GridDim.y = (img.rows + BlockDim.y - 1) / BlockDim.y;
    GridDim.z = 1;
    ImageBlur <<<GridDim, BlockDim>>> (d_in, d_out, img.cols, img.rows);

    // Wait until cudaKernel will finsh calculation
    cudaDeviceSynchronize();

    // Transfer Results to Host
    cudaMemcpy(result.data, d_out, SizeInBytes, cudaMemcpyDeviceToHost);

    // Save results
    cv::imwrite(output_path, result);

    // Reallocate Host and device Memory
    img.release();
    result.release();
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}