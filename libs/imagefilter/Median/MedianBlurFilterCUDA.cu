#include "MedianBlurFilterCUDA.hpp"
#include <stdio.h>

// TO DO: Improve sorting
__device__ void sortCUDA(int filter_width, int filter_height, unsigned char* filterVector)
{
    for (int i = 0; i < filter_width*filter_height; i++) {
        for (int j = i + 1; j < filter_width*filter_height; j++) {
            if (filterVector[i] > filterVector[j]) { 
                unsigned char tmp = filterVector[i];
                filterVector[i] = filterVector[j];
                filterVector[j] = tmp;
            }
        }
    }
}

__global__ void medianBlurFilterKernel( unsigned char *srcImage,
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
            int size = filter_width * filter_height;
            // printf("Allocating: %d\n", filter_width * filter_height);
            unsigned char *filterVector = static_cast<unsigned char*>(malloc(size * sizeof(unsigned char)));
            // Loop inside the filter to average pixel values
            for(int ky=-filter_height/2; ky<=filter_height/2; ky++) {
                for(int kx=-filter_width/2; kx<=filter_width/2; kx++) {
                    // printf("assigning at: %d\n", ky*filter_width+kx);
                    filterVector[((ky*filter_width+kx) + size) % size] = srcImage[((y+ky)*width + (x+kx))*channel+c];
                }
            }
            // Sorting values of filter   
            sortCUDA(filter_width, filter_height, filterVector);
            dstImage[(y*width+x)*channel+c] =  filterVector[(filter_width*filter_height)/2];
            free(filterVector);
        }
    }
}

MedianBlurFilterCUDA::MedianBlurFilterCUDA(unsigned int filter_kernel_width, 
                                            unsigned int filter_kernel_height, 
                                            unsigned int cuda_block_size)
: AbstractFilterCUDA(filter_kernel_width, filter_kernel_height, cuda_block_size)
{
    cuda_kernel = &medianBlurFilterKernel;
}

MedianBlurFilterCUDA::MedianBlurFilterCUDA(const MedianBlurFilterCUDA& other)
: AbstractFilterCUDA(other)
{
    cuda_kernel = other.cuda_kernel;
}

MedianBlurFilterCUDA& MedianBlurFilterCUDA::operator=(const MedianBlurFilterCUDA& other) {
    if(this != &other){
        AbstractFilterCUDA::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
    }

    return *this;
}

MedianBlurFilterCUDA::~MedianBlurFilterCUDA(){
}