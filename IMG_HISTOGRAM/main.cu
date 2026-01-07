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
    // Read Image
    char *img_str = argv[1];
    cv::Mat h_img = cv::imread(img_str, cv::IMREAD_GRAYSCALE);

    // Launch histogram cuda Main
    unsigned int *h_hist;
    gray_histogram_main(h_img, &h_hist);

    // Calculate Hist in opencv
    cv::Mat cv_hist;
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

    return 0;

}