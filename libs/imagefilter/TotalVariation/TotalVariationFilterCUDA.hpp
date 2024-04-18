#ifndef TOTALVARIATION_FILTER_CUDA_HPP
#define TOTALVARIATION_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class TotalVariationFilterCUDA : public AbstractFilterCUDA
{
public:
    TotalVariationFilterCUDA(unsigned int filter_kernel_width,
                             unsigned int filter_kernel_height,
                             CUDARunTimeConfig cuda_run_time_config);
    TotalVariationFilterCUDA(const TotalVariationFilterCUDA &other);
    TotalVariationFilterCUDA &operator=(const TotalVariationFilterCUDA &other);
    ~TotalVariationFilterCUDA();

    bool read(std::string input);
    bool save(std::string output);
};


#endif