#include <stdio.h>
#include <stdlib.h>
#include "ImageProcessing.cuh"


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Error (argc !=2) : ./<exe> <path to input image> \n");
        return 1;
    }

    const char* img_path = argv[1];
    int CHANNEL = 3;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);

    if (img.empty())
    {
        printf("Failed to read img path: %s \n", img_path);
        return 1;
    }

    printf("Loaded image: %s  (%d x %d, %d channels)\n", img_path, img.cols, img.rows, img.channels());

    int width = img.cols;
    int height = img.rows;

    // Define output img
    cv::Mat gray_img(height, width, CV_8UC1);
    cv::Mat blur_img(height, width, CV_8UC1);

    // call convert to gary
    colorToGray(img, gray_img);

    // call Box Filter
    BoxFilter_device(gray_img, blur_img, 5);

    // Save results
    cv::imwrite("Images/lenna_gray.jpg", gray_img);
    cv::imwrite("Images/lenna_blur.jpg", blur_img);

    // Deallocate host mem
    img.release();
    gray_img.release();
    blur_img.release();

    return 0;
}