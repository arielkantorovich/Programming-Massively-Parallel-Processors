#pragma once
#include <cuda_runtime.h>


__global__
void MatrixMut_kernel(float *M, float *N, float *P, int n, int m, int l);

__global__
void MatVecMult_kernel(float*  __restrict__ A, const float*  __restrict__ B, const float*  __restrict__ C, int n);

__global__
void MatrixMut_1a(float *M, float *N, float *P, int n, int m, int l);

__global__
void MatrixMut_1b(float *M, float *N, float *P, int n, int m, int l);

void MatVecMult_device(float *h_A, float *h_B, float *h_C, int n);