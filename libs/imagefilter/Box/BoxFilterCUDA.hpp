#ifndef BOX_FILTER_CUDA_HPP
#define BOX_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class BoxFilterCUDA : public AbstractFilterCUDA {
public:
    BoxFilterCUDA ( unsigned int filter_kernel_width, 
                    unsigned int filter_kernel_height, 
                    unsigned int cuda_block_size);
    BoxFilterCUDA(const BoxFilterCUDA& other);
    BoxFilterCUDA& operator=(const BoxFilterCUDA& other);
    ~BoxFilterCUDA();
};


#endif