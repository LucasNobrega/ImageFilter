#ifndef BOX_FILTER_CUDA_HPP
#define BOX_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class BoxFilterCUDA : public AbstractFilterCUDA
{
public:
    BoxFilterCUDA(unsigned int filter_kernel_width,
                  unsigned int filter_kernel_height,
                  CUDARunTimeConfig cuda_run_time_config);
    BoxFilterCUDA(const BoxFilterCUDA &other);
    BoxFilterCUDA &operator=(const BoxFilterCUDA &other);
    ~BoxFilterCUDA();
};


#endif