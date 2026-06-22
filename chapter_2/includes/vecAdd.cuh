#pragma once
#include <cuda_runtime.h>

__global__ void addVecKernel(float *A, float *B, float *C, unsigned int n);
void addVecDevice(float *h_a, float *h_b, float *h_c, unsigned int n);
void CompareHostDeviceResults(float *h_a, float *d_a, unsigned int n);