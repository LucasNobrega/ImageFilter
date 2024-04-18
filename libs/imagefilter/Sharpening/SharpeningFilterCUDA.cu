#include "SharpeningFilterCUDA.hpp"
#include <stdio.h>

__global__ void sharpeningFilterKernel(unsigned char *srcImage,
                                       unsigned char *dstImage,
                                       int filter_width,
                                       int filter_height,
                                       unsigned int width,
                                       unsigned int height,
                                       int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Create kernel
    float **kernel = (float **)malloc(filter_height * sizeof(float *));
    for (int i = 0; i < filter_height; ++i)
    {
        kernel[i] = (float *)malloc(filter_width * sizeof(float));
    }
    for (int i = 0; i < filter_height; ++i)
    {
        for (int j = 0; j < filter_width; ++j)
        {
            kernel[i][j] = -1;
        }
    }
    kernel[filter_height / 2][filter_width / 2] = 9;

    // only threads inside image will write results
    // Check if the indices are within the bounds of the image
    if (x >= 0 && x < width && y >= 0 && y < height)
    {
        for (int c = 0; c < channel; c++)
        {
            // Sum of pixel values
            float sum = 0;
            // Loop inside the filter to average pixel values
            for (int ky = -filter_height / 2; ky <= filter_height / 2; ky++)
            {
                for (int kx = -filter_width / 2; kx <= filter_width / 2; kx++)
                {
                    // Check if the indices are within the bounds of the image
                    if ((x + kx) >= 0 && (x + kx) < width && (y + ky) >= 0 &&
                        (y + ky) < height)
                    {
                        float fl =
                            srcImage[((y + ky) * width + (x + kx)) * channel +
                                     c];
                        sum +=
                            fl *
                            kernel[((ky + filter_height / 2) + filter_height) %
                                   filter_height]
                                  [((kx + filter_width / 2) + filter_width) %
                                   filter_width];
                    }
                }
            }
            dstImage[(y * width + x) * channel + c] = sum;
        }
    }

    for (size_t i = 0; i < filter_height; i++)
    {
        free(kernel[i]);
    }
    free(kernel);
}

SharpeningFilterCUDA::SharpeningFilterCUDA(
    unsigned int filter_kernel_width,
    unsigned int filter_kernel_height,
    CUDARunTimeConfig cuda_run_time_config)
    : AbstractFilterCUDA(filter_kernel_width,
                         filter_kernel_height,
                         cuda_run_time_config)
{
    cuda_kernel = &sharpeningFilterKernel;
}

SharpeningFilterCUDA::SharpeningFilterCUDA(const SharpeningFilterCUDA &other)
    : AbstractFilterCUDA(other)
{
    cuda_kernel = other.cuda_kernel;
}

SharpeningFilterCUDA &SharpeningFilterCUDA::operator=(
    const SharpeningFilterCUDA &other)
{
    if (this != &other)
    {
        AbstractFilterCUDA::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

SharpeningFilterCUDA::~SharpeningFilterCUDA()
{
}
