#ifndef LAPLACIAN_FILTER_HPP
#define LAPLACIAN_FILTER_HPP

#include "AbstractFilter.hpp"


class LaplacianFilter : public AbstractFilter {
public:
    LaplacianFilter();
    LaplacianFilter(const LaplacianFilter& other);
    LaplacianFilter& operator=(const LaplacianFilter& other);
    ~LaplacianFilter();

    bool apply();

private:
    int desired_depth;
    int aperture_size;
    double scale;
    double delta;
};


#endif