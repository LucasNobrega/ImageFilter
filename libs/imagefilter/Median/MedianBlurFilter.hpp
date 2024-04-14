#ifndef MEDIAN_BLUR_FILTER_HPP
#define MEDIAN_BLUR_FILTER_HPP

#include "AbstractFilter.hpp"


class MedianBlurFilter : public AbstractFilter {
public:
    MedianBlurFilter();
    MedianBlurFilter(const MedianBlurFilter& other);
    MedianBlurFilter& operator=(const MedianBlurFilter& other);
    ~MedianBlurFilter();

    bool apply();

private:
    int kernel_size;
};


#endif