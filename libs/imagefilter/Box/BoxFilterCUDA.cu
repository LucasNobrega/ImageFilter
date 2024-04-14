#include "BoxFilterCUDA.hpp"
#include <stdio.h>

__global__ void boxFilterKernel(unsigned char *srcImage,
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
        for(int c=0 ; c<channel ; c++)   
        {
            // Sum of pixel values 
            float sum = 0;
            // Number of filter pixels 
            float kS = 0; 
            // Loop inside the filter to average pixel values
            for(int ky=-filter_height/2; ky<=filter_height/2; ky++) {
                for(int kx=-filter_width/2; kx<=filter_width/2; kx++) {
                    float fl = srcImage[((y+ky)*width + (x+kx))*channel+c];
                    sum += fl;
                    kS += 1;
                }
            }
            dstImage[(y*width+x)*channel+c] =  sum / kS;
        }
    }
}

BoxFilterCUDA::BoxFilterCUDA(   unsigned int filter_kernel_width,
                                unsigned int filter_kernel_height,
                                unsigned int cuda_block_size)
: AbstractFilterCUDA(filter_kernel_width, filter_kernel_height, cuda_block_size)
{
    cuda_kernel = &boxFilterKernel;
}

BoxFilterCUDA::BoxFilterCUDA(const BoxFilterCUDA& other)
: AbstractFilterCUDA(other)
{
    cuda_kernel = other.cuda_kernel;
}

BoxFilterCUDA& BoxFilterCUDA::operator=(const BoxFilterCUDA& other) {
    if(this != &other){
        AbstractFilterCUDA::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

BoxFilterCUDA::~BoxFilterCUDA(){
}