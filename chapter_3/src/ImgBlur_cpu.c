#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>


/**
 * @brief Implement image processing box filter average on kernel size.
 * @param Pin Input image
 * @param Pout Output image
 * @param KERNEL_SIZE - kernel size equal to 2R+1
 * @param width
 * @param height
 * @note assume input u_char8 and channel equal to 1.
 */
void BoxFilter(unsigned char *Pin, unsigned char *Pout, int KERNEL_SIZE, int width, int height)
{
    int RADIUS = KERNEL_SIZE / 2;
    for (int Row=0; Row < height; Row++)
    {
        for (int Col=0; Col < width; Col++)
        {
            int sum = 0;
            int count = 0;
            for (int fRow=0; fRow < KERNEL_SIZE; fRow++)
            {
                for (int fCol=0; fCol < KERNEL_SIZE; fCol++)
                {
                    int c = Col + fCol - RADIUS;
                    int r = Row + fRow - RADIUS;
                    if (c >= 0 && c < width && r >= 0 && r < height)
                    {
                        sum += Pin[c + r * width];
                        count++;
                    }
                }
            }
            Pout[Col + Row * width] = (unsigned char)(sum / count);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: ./<exe> <ImagePath>\n");
        return -1;
    }
    // Read Image as gray
    char *img_path = argv[1];
    cv::Mat gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    
    if (gray.empty())
    {
        printf("Error: read empty image %s \n", argv[1]);
        gray.release();
        return -1;
    }

    // Define parameters
    int width = gray.cols;
    int height = gray.rows;
    int KERNEL_SIZE = 5;

    printf("Loaded image: %s  (%d x %d, %d channels)\n", img_path, height, width, gray.channels());
    
    // Define output image
    cv::Mat blur(height, width, CV_8UC1);

    // Call Box Filter
    BoxFilter(gray.data, blur.data, KERNEL_SIZE, width, height);

    // Save results
    cv::imwrite("Images/lenna_blur_cpu.jpg", blur);

    // Deallocate memo
    blur.release();
    gray.release();

    return 0;
}