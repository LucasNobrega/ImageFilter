#ifndef SHARPENING_FILTER_CUDA_HPP
#define SHARPENING_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class SharpeningFilterCUDA : public AbstractFilterCUDA {
public:
    SharpeningFilterCUDA(unsigned int filter_kernel_width,
                         unsigned int filter_kernel_height,
                         unsigned int cuda_block_size);
    SharpeningFilterCUDA(const SharpeningFilterCUDA& other);
    SharpeningFilterCUDA& operator=(const SharpeningFilterCUDA& other);
    ~SharpeningFilterCUDA();
};


#endif