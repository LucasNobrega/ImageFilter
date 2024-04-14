#include "LaplacianFilter.hpp"

LaplacianFilter::LaplacianFilter()
    : AbstractFilter(), desired_depth(0), aperture_size(5), scale(1), delta(0)
{
}

LaplacianFilter::LaplacianFilter(const LaplacianFilter &other)
    : AbstractFilter(other), desired_depth(other.desired_depth),
      aperture_size(other.aperture_size), scale(other.scale), delta(other.delta)
{
}

LaplacianFilter &LaplacianFilter::operator=(const LaplacianFilter &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
        this->desired_depth = other.desired_depth;
        this->aperture_size = other.aperture_size;
        this->scale = other.scale;
        this->delta = other.delta;
    }

    return *this;
}

LaplacianFilter::~LaplacianFilter()
{
}

bool LaplacianFilter::apply()
{
    if (raw_image->empty())
    {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::Laplacian(*raw_image,
                  *filtered_image,
                  desired_depth,
                  aperture_size,
                  scale,
                  delta);
    return true;
}
