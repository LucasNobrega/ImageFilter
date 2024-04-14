#ifndef LAPLACIAN_FILTER_CUDA_HPP
#define LAPLACIAN_FILTER_CUDA_HPP

#include "AbstractFilterCUDA.hpp"

class LaplacianFilterCUDA : public AbstractFilterCUDA {
public:
    LaplacianFilterCUDA(int filter_kernel_width,
                        int filter_kernel_height,
                        int depth,
                        int aperture,
                        double scale,
                        double delta,
                        int cuda_block_size);

    LaplacianFilterCUDA(const LaplacianFilterCUDA& other);
    LaplacianFilterCUDA& operator=(const LaplacianFilterCUDA& other);
    ~LaplacianFilterCUDA();

private:
    int depth;
    int aperture;
    double scale;
    double delta;
};


#endif