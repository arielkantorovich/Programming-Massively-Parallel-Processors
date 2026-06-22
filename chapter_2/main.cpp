#include <stdio.h>
#include <stdlib.h>
#include "vecAdd.hpp"
#include "vecAdd.cuh"

int main(void)
{
    unsigned int n = 1000;
    // allocate host memory
    float *h_a = (float*)malloc(sizeof(float) * n);
    float *h_b = (float*)malloc(sizeof(float) * n);
    float *h_c = (float*)malloc(sizeof(float) * n);
    float *hd_c = (float*)malloc(sizeof(float) * n);
    
    // Set Number in a vector
    SetNumInVec(h_a, 1, n);
    SetNumInVec(h_b, 2, n);
    
    // Call host kernel
    addVec(h_a, h_b, h_c, n);
    PrintVecElem(h_c, n);

    // Call Device kernel
    addVecDevice(h_a, h_b, hd_c, n);

    // Compare results
    CompareHostDeviceResults(h_c, hd_c, n);

    // Dellocate Memory 
    free(h_a);
    free(h_b);
    free(h_c);
    free(hd_c);

    return 0;
}