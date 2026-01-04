#include <stdio.h>
#include <stdlib.h>

void printResults(unsigned int* hist)
{
    const char *x[7] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int i=0; i < 7; i++)
    {
        printf("%s = %i \n", x[i], hist[i]);
    }
}

void histogram_sequential(const char *seq, unsigned int length, unsigned int* hist)
{
    for (int i=0; i < length; i++)
    {
        int alphabet_position = seq[i] - 'a';
        if (alphabet_position >=0 && alphabet_position < 26)
        {
            hist[alphabet_position/4] +=1;
        }
    }
}


int main(int argc, char **argv)
{
    // Define sequence vector
    const char *seq = "programming massively parallel processors";
    unsigned int N = 42;
    // Define histogram size ceil(26 / 4) where 26-number of letters and 4 
    // letter in each group
    unsigned int hist[7] = {0};
    histogram_sequential(seq, N, hist);
    printResults(hist);
    return 0;
}