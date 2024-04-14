#include "BoxFilter.hpp"

BoxFilter::BoxFilter()
: AbstractFilter()
, kernel_size(11)
{
}

BoxFilter::BoxFilter(const BoxFilter& other)
: AbstractFilter(other)
, kernel_size(other.kernel_size)
{
}

BoxFilter& BoxFilter::operator=(const BoxFilter& other) {
    if(this != &other){
        AbstractFilter::operator=(other);
        this->kernel_size = other.kernel_size;
    }

    return *this;
}

BoxFilter::~BoxFilter(){
}

bool BoxFilter::apply()
{
    if (raw_image->empty()) {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::boxFilter(*raw_image, *filtered_image, -1, cv::Size(kernel_size, kernel_size));
    return true;
}

