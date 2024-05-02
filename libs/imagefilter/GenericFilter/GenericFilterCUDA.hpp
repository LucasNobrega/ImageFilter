#ifndef GENERIC_FILTER_CUDA_HPP
#define GENERIC_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"
#include <fstream>

class GenericFilterCUDA : public AbstractFilterCUDA
{
public:
    GenericFilterCUDA(CUDARunTimeConfig cuda_run_time_config);
    GenericFilterCUDA(const GenericFilterCUDA &other);
    GenericFilterCUDA &operator=(const GenericFilterCUDA &other);
    ~GenericFilterCUDA();
    
    bool configureKernel(const std::string& filename);

    typedef void (*generic_cuda_kernel_ptr)(unsigned char *,
                                            unsigned char *,
                                            unsigned int,
                                            unsigned int,
                                            float*,
                                            int,
                                            int);

private:
    cv::Mat kernel;
    void activate(cv::Mat *input, cv::Mat *output);

protected:
    generic_cuda_kernel_ptr generic_cuda_kernel;
};


#endif