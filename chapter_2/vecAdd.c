#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Add vec on host h_a+h_b = h_c
 * @param n - size of vectors
*/
void addVec(float *h_a, float *h_b, float *h_c, int n)
{
    for (int i=0; i<n; i++)
    {
        h_c[i] = h_a[i] + h_b[i];
    }
}

int main(void)
{   
    // allocate memory in host
    int size_arr = 1000;
    int size_bytes = sizeof(float) * size_arr;

    float *a = (float*)malloc(size_bytes);
    float *b = (float*)malloc(size_bytes);
    float *c = (float*)malloc(size_bytes);

    for (int i = 0; i < size_arr; i++)
    {
        a[i] = 0.1f;
        b[i] = 2.2f;
        c[i] = 0.0f;
    }

    addVec(a, b, c, size_arr);

    for (int i=0; i < size_arr; i++)
    {
        printf("c[%i]=%f \n", i, c[i]);
    }


    // Dellocate memory
    free(a);
    free(b);
    free(c);

    return 0;
}