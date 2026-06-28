#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void colorToGrayKernel(unsigned char *Pin, unsigned char *Pout, int width, int height, int CHANNEL=3);
void colorToGray(cv::Mat Pin, cv::Mat Pout);
__global__ void BoxFilter_kernel(unsigned char *Pin, unsigned char *Pout, int KERNEL_SIZE, int width, int height);
void BoxFilter_device(cv::Mat gray, cv::Mat blur, int KERNEL_SIZE);