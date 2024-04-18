#ifndef SHARPENING_FILTER_CUDA_HPP
#define SHARPENING_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class SharpeningFilterCUDA : public AbstractFilterCUDA
{
public:
    SharpeningFilterCUDA(unsigned int filter_kernel_width,
                         unsigned int filter_kernel_height,
                         CUDARunTimeConfig cuda_run_time_config);
    SharpeningFilterCUDA(const SharpeningFilterCUDA &other);
    SharpeningFilterCUDA &operator=(const SharpeningFilterCUDA &other);
    ~SharpeningFilterCUDA();
};


#endif