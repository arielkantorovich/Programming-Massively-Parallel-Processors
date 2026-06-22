#include "vecAdd.hpp"
#include <stdio.h>

/**
 * @brief Implement vector addition element-wise on host.
 * @param A Input vector size n.
 * @param B Input vector size n.
 * @param C Output vector size n.
 */
void addVec(float *A, float *B, float *C, unsigned int n)
{
    for (unsigned int i=0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}


/**
 * @brief Set constant number at each element in vector.
 * @param A - Input vector
 * @param num - number
 * @param n - vector length 
 */
void SetNumInVec(float* A, int num, unsigned int n)
{
    for (unsigned int i=0; i < n; i++)
    {
        A[i] = num;
    }
}

/**
 * @param A - Input vec
 * @param n - vector length
 */
void PrintVecElem(float* A, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        printf("A[%i]=%f \n", i, A[i]);
    }
}