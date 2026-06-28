#include "ImageProcessing.cuh"

/**
 * @brief The following kernel method convert RGB image to gray.
 * @param Pin - input image HxWxCHANNEl where CHANNEL (3)
 * @param Pout - gray output image CHANNEL (1)
 * @note  Because we read opencv matrix that is BGR format
 * the channel offset is swep between red and blue.
 * 
*/
__global__
void colorToGrayKernel(unsigned char *Pin, unsigned char *Pout, int width, int height, int CHANNEL)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if ((Row < height) && (Col < width))
    {
        int gridOffset =  Col + Row * width;
        int channelOfsset = CHANNEL * gridOffset;
        unsigned char b = Pin[channelOfsset];
        unsigned char g = Pin[channelOfsset + 1];
        unsigned char r = Pin[channelOfsset + 2];
        Pout[gridOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}



/**
 * @brief convert BGR image to gray.
 * @param Pin input image bgr type unsigned char.
 * @param Pout output image gray type unsigned char one channel.
 */
void colorToGray(cv::Mat Pin, cv::Mat Pout)
{
    if (Pin.empty())
    {
        printf("colorToGray func: input image is empty");
        return;
    }
    
    int width = Pin.cols;
    int height = Pin.rows;
    int CHANNEL = Pin.channels();

    // allocate device Memory
    unsigned char *d_gray, *d_bgr;
    cudaMalloc((void**)&d_bgr, sizeof(unsigned char) * width * height * CHANNEL);
    cudaMalloc((void**)&d_gray, sizeof(unsigned char) * width * height);

    // Transfer data from host to device
    cudaMemcpy(d_bgr, Pin.data, sizeof(unsigned char) * width * height * CHANNEL, cudaMemcpyHostToDevice);

    // Define grid size
    dim3 dimThread(16, 16, 1);
    dim3 dimGrid;
    dimGrid.x = (width - 1 + dimThread.x) / dimThread.x;
    dimGrid.y = (height - 1 + dimThread.y) / dimThread.y;

    // Launch kernel
    colorToGrayKernel <<<dimGrid, dimThread>>>(d_bgr, d_gray, width, height, CHANNEL);
    cudaDeviceSynchronize();

    // Transfer results to host
    cudaMemcpy(Pout.data, d_gray, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(d_bgr);
    cudaFree(d_gray);

}


/**
 * @brief Box filter implementation on device
 * @param Pin Input image
 * @param Pout Output image
 * @param KERNEL_SIZE - kernel size equal to 2R+1
 * @param width
 * @param height
 * @note assume input u_char8 and channel equal to 1.
 */
__global__
void BoxFilter_kernel(unsigned char *Pin, unsigned char *Pout, int KERNEL_SIZE, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (Col < width && Row < height)
    {
        int RADIUS = KERNEL_SIZE / 2;
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



/**
 * @brief BoxFilter host area
 * @param gray - input image
 * @param blur - output image
 * @param KERNEL_SIZE 
 */
void BoxFilter_device(cv::Mat gray, cv::Mat blur, int KERNEL_SIZE)
{
    
    if (gray.empty())
    {
        printf("BoxFilter_device func: input image is empty");
        return;
    }

    int width = gray.cols;
    int height = gray.rows;

    // allocate device Memory
    unsigned char *d_gray, *d_blur;
    cudaMalloc((void**)&d_gray, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&d_blur, sizeof(unsigned char) * width * height);

    // Transfer data from host to device
    cudaMemcpy(d_gray, gray.data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // Define grid size
    dim3 dimThread(16, 16, 1);
    dim3 dimGrid;
    dimGrid.x = (width - 1 + dimThread.x) / dimThread.x;
    dimGrid.y = (height - 1 + dimThread.y) / dimThread.y;

    // Launch kernel
    BoxFilter_kernel <<<dimGrid, dimThread>>>(d_gray, d_blur, KERNEL_SIZE, width, height);
    cudaDeviceSynchronize();

    // Transfer results to host
    cudaMemcpy(blur.data, d_blur, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(d_gray);
    cudaFree(d_blur);
}