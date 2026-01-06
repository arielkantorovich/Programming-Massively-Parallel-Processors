#include "histogram_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

/**
 * @brief Compare between the techniques with cuda and histogram CPU of opencv
 * @param histCUDA results from cuda kernel on the host
 * @param histCV results from calcHist of opencv in the host.
 * @note calcHist return cv::Mat with type CV_F32 which is float
 */
void CompareHist(unsigned int* histCUDA, float* histCV)
{
    for (unsigned int i=0; i < NUM_BINS; i++)
    {
        if (histCUDA[i] != static_cast<unsigned int>(histCV[i]))
        {
            printf("Get Diffrent values histCUDA[%i]=%i != histCV[%i]=%i \n", i, histCUDA[i], i, (unsigned int)histCV[i]);
            return ;
        }
    }
    printf("The results is correct...\n");
    return;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error: runing command: ./<exe> <image_path>\n");
        return -1;
    }

    char *img_str = argv[1];
    cv::Mat h_img = cv::imread(img_str, cv::IMREAD_GRAYSCALE);
    unsigned int NUM_PIXELS = static_cast<unsigned int>(h_img.cols * h_img.rows);
    

    // Allocate host memory
    size_t SizeInBytes_hist = sizeof(unsigned int) * NUM_BINS;
    size_t SizeInBytes_img = sizeof(unsigned char) * NUM_PIXELS;

    unsigned int *h_hist;
    cv::Mat cv_hist;
    cudaMallocHost((void**)&h_hist, SizeInBytes_hist);

    // Allocate device Memory
    unsigned char* d_img;
    unsigned int* d_hist;

    cudaMalloc((void**)&d_img, SizeInBytes_img);
    cudaMalloc((void**)&d_hist, SizeInBytes_hist);

    // Copy data from host to device
    cudaMemcpy(d_img, h_img.data, SizeInBytes_img, cudaMemcpyHostToDevice);

    // Define Grid Size
    dim3 NumThreadPerBlock(256, 1, 1);
    dim3 GridSize;
    GridSize.z = 1;
    GridSize.y = 1;
    GridSize.x = (NUM_PIXELS + NumThreadPerBlock.x - 1) / NumThreadPerBlock.x;

    gray_hist_aggregate_kernel <<<GridSize, NumThreadPerBlock>>> (d_img, NUM_PIXELS, d_hist);
    cudaDeviceSynchronize();

    // Move data from device to host
    cudaMemcpy(h_hist, d_hist, SizeInBytes_hist, cudaMemcpyDeviceToHost);

    // Calculate Hist in opencv
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    int channels[] = { 0 };
    cv::calcHist(&h_img, 1, channels, cv::Mat(), cv_hist, 1, &histSize, histRange);

    // Compare results
    CompareHist(h_hist, cv_hist.ptr<float>());

    // Free Host and Device Memory
    cudaFreeHost(h_hist);
    h_img.release();
    cudaFree(d_img);
    cudaFree(d_img);

    return 0;

}