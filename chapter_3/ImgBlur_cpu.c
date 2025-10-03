#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define KERNEL_SIZE 5
#define BLUR_RADIUS ((KERNEL_SIZE) / 2)

/**
 * @brief Box Filter implementation on CPU
 * @param Pin - input image
 * @param Pout - output image
*/
void ImgBlur(unsigned char *Pin, unsigned char *Pout, int n)
{
    for (int Row=0; Row < n; Row++)
    {
        for (int Col=0; Col < n; Col++)
        {
            int sum = 0;
            int count = 0;
            
            for (int K_row=-BLUR_RADIUS; K_row < BLUR_RADIUS+1; K_row++)
            {
                for (int K_col=-BLUR_RADIUS; K_col < BLUR_RADIUS+1; K_col++)
                {
                    int R = Row + K_row;
                    int C = Col + K_col;
                    if ((R > -1 && R < n) && (C > -1 && C < n))
                    {
                        sum += Pin[C + R*n];
                        count ++;
                    }
                }
            }
            Pout[Col + Row*n] = (unsigned char)(sum / count);
        }
    }
}


int main(void)
{
    const char* input_path = "Images/lenna.jpg";
    const char* output_path = "Images/lenna_blur.jpg";

    cv::Mat gray = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (gray.empty())
    {
        printf("Not sucssed to open image in the following path: %s", input_path);
        return 1;
    }

    // Alocate results Memory
    cv::Mat blur(gray.rows, gray.cols, gray.type());

    // Launch CPU Method
    ImgBlur(gray.data, blur.data, gray.cols);

    // Save results
    cv::imwrite(output_path, blur);

    // release CPU Memo
    blur.release();
    gray.release();

    return 0;
}