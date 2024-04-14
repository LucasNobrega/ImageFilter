#include "MedianBlurFilter.hpp"

MedianBlurFilter::MedianBlurFilter()
: AbstractFilter()
, kernel_size(11)
{
}

MedianBlurFilter::MedianBlurFilter(const MedianBlurFilter& other)
: AbstractFilter(other)
, kernel_size(other.kernel_size)
{
}

MedianBlurFilter& MedianBlurFilter::operator=(const MedianBlurFilter& other) {
    if(this != &other){
        AbstractFilter::operator=(other);
        this->kernel_size = other.kernel_size;
    }

    return *this;
}

MedianBlurFilter::~MedianBlurFilter(){
}

bool MedianBlurFilter::apply()
{
    if (raw_image->empty()) {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::medianBlur(*raw_image, *filtered_image, kernel_size);
    return true;
}

