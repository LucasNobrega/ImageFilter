#include "SobelEdgeDetectionFilterCUDA.hpp"
#include <stdio.h>

__global__ void sobelEdgeDetectionFilterKernel(unsigned char *srcImage,
                                               unsigned char *dstImage,
                                               int filter_width,
                                               int filter_height,
                                               unsigned int width,
                                               unsigned int height,
                                               int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float Ky[3][3] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    // only threads inside image will write results
    if ((x >= filter_width / 2) && (x < (width - filter_width / 2)) &&
        (y >= filter_height / 2) && (y < (height - filter_height / 2)))
    {
        // Gradient in x-direction
        float Gx = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -filter_height / 2; ky <= filter_height / 2; ky++)
        {
            for (int kx = -filter_width / 2; kx <= filter_width / 2; kx++)
            {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gx += fl * Kx[ky + filter_height / 2][kx + filter_width / 2];
            }
        }
        float Gx_abs = Gx < 0 ? -Gx : Gx;

        // Gradient in y-direction
        float Gy = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -filter_height / 2; ky <= filter_height / 2; ky++)
        {
            for (int kx = -filter_width / 2; kx <= filter_width / 2; kx++)
            {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gy += fl * Ky[ky + filter_height / 2][kx + filter_width / 2];
            }
        }
        float Gy_abs = Gy < 0 ? -Gy : Gy;

        dstImage[(y * width + x)] = Gx_abs + Gy_abs;
    }
}

SobelEdgeDetectionFilterCUDA::SobelEdgeDetectionFilterCUDA(
    unsigned int filter_kernel_width,
    unsigned int filter_kernel_height,
    CUDARunTimeConfig cuda_run_time_config)
    : AbstractFilterCUDA(filter_kernel_width,
                         filter_kernel_height,
                         cuda_run_time_config)
{
    cuda_kernel = &sobelEdgeDetectionFilterKernel;
}

SobelEdgeDetectionFilterCUDA::SobelEdgeDetectionFilterCUDA(
    const SobelEdgeDetectionFilterCUDA &other)
    : AbstractFilterCUDA(other)
{
    cuda_kernel = other.cuda_kernel;
}

SobelEdgeDetectionFilterCUDA &SobelEdgeDetectionFilterCUDA::operator=(
    const SobelEdgeDetectionFilterCUDA &other)
{
    if (this != &other)
    {
        AbstractFilterCUDA::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

SobelEdgeDetectionFilterCUDA::~SobelEdgeDetectionFilterCUDA()
{
}

bool SobelEdgeDetectionFilterCUDA::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }

    cv::cvtColor(*raw_image, *raw_image, cv::COLOR_BGR2GRAY);
    filtered_image->create(raw_image->size(), raw_image->type());

    return true;
}
