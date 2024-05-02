#ifndef ABSTRACT_FILTER_CUDA_HPP
#define ABSTRACT_FILTER_CUDA_HPP

#include "AbstractFilter.hpp"
#include "CUDARunTimeConfig.hpp"

class AbstractFilterCUDA : public AbstractFilter
{
public:
    AbstractFilterCUDA(unsigned int filter_kernel_width,
                       unsigned int filter_kernel_height,
                       CUDARunTimeConfig cuda_run_time_config);
    AbstractFilterCUDA(const AbstractFilterCUDA &other);
    AbstractFilterCUDA &operator=(const AbstractFilterCUDA &other);
    ~AbstractFilterCUDA();

    bool apply();
    bool read(std::string input) override;

    typedef void (*cuda_kernel_ptr)(unsigned char *,
                                    unsigned char *,
                                    int,
                                    int,
                                    unsigned int,
                                    unsigned int,
                                    int);

protected:
    unsigned int filter_kernel_width;
    unsigned int filter_kernel_height;
    virtual void activate(cv::Mat *input, cv::Mat *output);
    virtual void configGrid();
    virtual bool checkCUDAConfig();
    CUDARunTimeConfig cuda_run_time_config;
    cuda_kernel_ptr cuda_kernel;
};


#endif