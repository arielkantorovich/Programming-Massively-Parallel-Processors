#include <stdio.h>
#include <stdlib.h>


/**
 * @brief Implement 2D-convolution on CPU device O(N^2 * K^2)
 * @param In - input matrix size (height, width)
 * @param F - filter (kernel) size (2r+1, 2r+1)
 * @param Out - output image same size as input
 * @param (r, width, height) specify size
*/
void conv2D(float *In, float *F, float *Out, unsigned int r, unsigned int width, unsigned int height)
{
    int KERNEL_SIZE = 2 * r + 1;

    for (int Row=0; Row < height; Row++)
    {
        for (int Col=0; Col < width; Col++)
        {
            float Pval = 0.0f;
            
            for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
            {
                for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
                {
                    int InRow = Row - r + fRow;
                    int InCol = Col - r + fCol;
                    if (((InRow >= 0) && (InRow < height))  &&  ((InCol >= 0) && (InCol < width)))
                    {
                        Pval += In[InCol + InRow * width] * F[fCol + fRow * KERNEL_SIZE];
                    }
                }
            }
            Out[Col + Row * width] = Pval;
        }
    }
}


int main(int argc, char **argv)
{
    int r = 1;
    float kernel[] = {
                        1.0f, 2.0f, 1.0f,
                        2.0f, 4.0f, 2.0f,
                        1.0f, 2.0f, 1.0f
                        };

    return 0;
}