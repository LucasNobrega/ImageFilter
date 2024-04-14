#ifndef ABSTRACT_FILTER_CUDA_HPP
#define ABSTRACT_FILTER_CUDA_HPP

#include "AbstractFilter.hpp"

class AbstractFilterCUDA : public AbstractFilter {
public:
    AbstractFilterCUDA( unsigned int filter_kernel_width, 
                        unsigned int filter_kernel_height, 
                        unsigned int cuda_block_size);
    AbstractFilterCUDA(const AbstractFilterCUDA& other);
    AbstractFilterCUDA& operator=(const AbstractFilterCUDA& other);
    ~AbstractFilterCUDA();

    bool apply();
    bool read(std::string input) override;

    typedef void (*cuda_kernel_ptr)(  unsigned char*
                                    , unsigned char*
                                    , int
                                    , int
                                    , unsigned int
                                    , unsigned int
                                    , int);

private:
    unsigned int filter_kernel_width;
    unsigned int filter_kernel_height;
    unsigned int cuda_block_size;
    
    virtual void activate(cv::Mat* input, cv::Mat* output);

protected:
    cuda_kernel_ptr cuda_kernel;
};


#endif