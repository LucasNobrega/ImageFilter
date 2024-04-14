#include "TotalVariationFilterCUDA.hpp"
#include <stdio.h>

__global__ void totalVariationFilterKernel(unsigned char *srcImage,
                                           unsigned char *dstImage,
                                           int filter_width,
                                           int filter_height,
                                           unsigned int width,
                                           unsigned int height,
                                           int channel)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // only threads inside image will write results
    if((x>=filter_width/2) && (x<(width-filter_width/2)) && (y>=filter_height/2) && (y<(height-filter_height/2)))
    {
            float sod = 0;
            // Loop inside the filter to average pixel values
            for(int ky=-filter_height/2; ky<=filter_height/2; ky++) {
                for(int kx=-filter_width/2; kx<=filter_width/2; kx++) {
                    float fl = srcImage[((y+ky)*width + (x+kx))];
                    float center = srcImage[((y)*width + (x))];
                    sod += fl-center;
                }
            }
            dstImage[(y*width+x)] = sod;
    }
}

TotalVariationFilterCUDA::TotalVariationFilterCUDA(unsigned int filter_kernel_width,
                                                   unsigned int filter_kernel_height,
                                                   unsigned int cuda_block_size)
: AbstractFilterCUDA(filter_kernel_width, filter_kernel_height, cuda_block_size)
{
    cuda_kernel = &totalVariationFilterKernel;
}

TotalVariationFilterCUDA::TotalVariationFilterCUDA(const TotalVariationFilterCUDA& other)
: AbstractFilterCUDA(other)
{
    cuda_kernel = other.cuda_kernel;
}

TotalVariationFilterCUDA& TotalVariationFilterCUDA::operator=(const TotalVariationFilterCUDA& other) {
    if(this != &other){
        AbstractFilterCUDA::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

TotalVariationFilterCUDA::~TotalVariationFilterCUDA(){
}

bool TotalVariationFilterCUDA::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }
    cv::cvtColor(*raw_image, *raw_image, cv::COLOR_BGR2GRAY);

    filtered_image->create(raw_image->size(), raw_image->type());

    return true;
}

bool TotalVariationFilterCUDA::save(std::string output)
{
    if (filtered_image->empty()) {
        std::cerr << "Error: image not filtered" << std::endl;
        return false;
    }
    filtered_image->convertTo(*filtered_image, CV_32F, 1.0 / 255, 0);
    (*filtered_image)*=255;

    if (cv::imwrite(output, *filtered_image)) {
        std::cerr << "Error: Failed to save the image" << std::endl;
        return false;
    }

    return true;
}

