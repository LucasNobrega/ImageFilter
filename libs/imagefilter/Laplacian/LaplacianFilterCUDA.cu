#include "LaplacianFilterCUDA.hpp"
#include <stdio.h>

__global__ void laplacianFilterKernel(unsigned char *srcImage,
                                      unsigned char *dstImage,
                                      int filter_width,
                                      int filter_height,
                                      unsigned int width,
                                      unsigned int height,
                                      int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float kernel[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    // only threads inside image will write results
    if ((x >= filter_width / 2) && (x < (width - filter_width / 2)) &&
        (y >= filter_height / 2) && (y < (height - filter_height / 2)))
    {
        for (size_t c = 0; c < 3; c++)
        {
            // Sum of pixel values
            float sum = 0;
            // Loop inside the filter to average pixel values
            for (int ky = -filter_height / 2; ky <= filter_height / 2; ky++)
            {
                for (int kx = -filter_width / 2; kx <= filter_width / 2; kx++)
                {
                    float fl = srcImage[((y + ky) * width + (x + kx)) * 3 + c];
                    sum +=
                        fl *
                        kernel[ky + filter_height / 2][kx + filter_width / 2];
                }
            }
            dstImage[(y * width + x) * 3 + c] = sum;
        }
    }
}

LaplacianFilterCUDA::LaplacianFilterCUDA(int filter_kernel_width,
                                         int filter_kernel_height,
                                         int depth,
                                         int aperture,
                                         double scale,
                                         double delta,
                                         CUDARunTimeConfig cuda_run_time_config)
    : AbstractFilterCUDA(filter_kernel_width,
                         filter_kernel_height,
                         cuda_run_time_config),
      depth(depth), aperture(aperture), scale(scale), delta(delta)
{
    cuda_kernel = &laplacianFilterKernel;
}

LaplacianFilterCUDA::LaplacianFilterCUDA(const LaplacianFilterCUDA &other)
    : AbstractFilterCUDA(other), depth(other.depth), aperture(other.aperture),
      scale(other.scale), delta(other.delta)
{
    cuda_kernel = other.cuda_kernel;
}

LaplacianFilterCUDA &LaplacianFilterCUDA::operator=(
    const LaplacianFilterCUDA &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
        this->depth = other.depth;
        this->aperture = other.aperture;
        this->scale = other.scale;
        this->delta = other.delta;
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

LaplacianFilterCUDA::~LaplacianFilterCUDA()
{
}
