#pragma once
#include <stdio.h>

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1


/**
 * @brief Put constant inside the 3D structure
 * @param in - 3D structure that we put constant inside
 * @param c - a constant flaot
 * @param n - dim in each axes.
*/
void putConst(float *in, float c, int n)
{
    for (int i = 0; i < n * n * n; i++)
    {
        in[i] = c;
    }
}


void PrintResult(float *out, int n)
{
    for (int i=0; i < n * n * n; i++)
    {
        printf("Out[%i] = %f \n", i, out[i]);
    }
}