#ifndef SOBELEDGEDETECTION_FILTER_CUDA_HPP
#define SOBELEDGEDETECTION_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class SobelEdgeDetectionFilterCUDA : public AbstractFilterCUDA {
public:
    SobelEdgeDetectionFilterCUDA(unsigned int filter_kernel_width,
                                 unsigned int filter_kernel_height,
                                 unsigned int cuda_block_size);
    SobelEdgeDetectionFilterCUDA(const SobelEdgeDetectionFilterCUDA& other);
    SobelEdgeDetectionFilterCUDA& operator=(const SobelEdgeDetectionFilterCUDA& other);
    ~SobelEdgeDetectionFilterCUDA();

    bool read(std::string input);
};


#endif