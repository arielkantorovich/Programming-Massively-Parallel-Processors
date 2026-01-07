#pragma once
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include <stdint.h>
#include <stdio.h>


/* ===============================
   Compile-time configuration
   =============================== */

#ifndef NUM_BINS
#define NUM_BINS 256
#endif

#ifndef BIN_SHIFT
#define BIN_SHIFT 0
#endif

static_assert(NUM_BINS > 0, "NUM_BINS must be positive");
static_assert((256 >> BIN_SHIFT) == NUM_BINS,
              "BIN_SHIFT inconsistent with NUM_BINS");

/* ===============================
   Kernel declarations
   =============================== */

__global__
void gray_hist_aggregate_kernel(const unsigned char* __restrict__ data,
                                unsigned int num_pixels,
                                unsigned int* __restrict__ histo);

__host__
void gray_histogram_main(cv::Mat img, unsigned int** h_hist);


__global__
void bgr_hist_aggregate_kernel(const unsigned char* __restrict__ data,
                                unsigned int num_pixels,
                                unsigned int* __restrict__ histo);