#ifndef MEDIAN_BLUR_FILTER_CUDA_HPP
#define MEDIAN_BLUR_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class MedianBlurFilterCUDA : public AbstractFilterCUDA {
public:
    MedianBlurFilterCUDA(unsigned int filter_kernel_width, 
                         unsigned int filter_kernel_height, 
                         unsigned int cuda_block_size);
    MedianBlurFilterCUDA(const MedianBlurFilterCUDA& other);
    MedianBlurFilterCUDA& operator=(const MedianBlurFilterCUDA& other);
    ~MedianBlurFilterCUDA();
};


#endif