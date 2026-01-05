#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>


__global__
void gray_histo_kernel(const unsigned char* __restrict__ img,
                        unsigned int * __restrict__ histo,
                        int width, int height)
                        {
                            int a;
                        }