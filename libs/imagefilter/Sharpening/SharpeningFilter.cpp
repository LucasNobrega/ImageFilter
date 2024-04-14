#include "SharpeningFilter.hpp"

SharpeningFilter::SharpeningFilter() : AbstractFilter()
{
}

SharpeningFilter::SharpeningFilter(const SharpeningFilter &other)
    : AbstractFilter(other)
{
}

SharpeningFilter &SharpeningFilter::operator=(const SharpeningFilter &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
    }

    return *this;
}

SharpeningFilter::~SharpeningFilter()
{
}

bool SharpeningFilter::apply()
{
    if (raw_image->empty())
    {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::Point anchor = cv::Point(-1, -1);
    double delta = 0;
    int ddepth = 0;
    int kernel_size;


    /// Update kernel size for a normalized box filter
    kernel_size = 3;

    cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) << -1,
                      -1,
                      -1,
                      -1,
                      9,
                      -1,
                      -1,
                      -1,
                      -1);

    // Apply 2D filter to image
    cv::filter2D(*raw_image, *filtered_image, ddepth, kernel, anchor, delta);

    return true;
}
