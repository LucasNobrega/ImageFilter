#include "AbstractFilter.hpp"

AbstractFilter::AbstractFilter()
    : raw_image(new cv::Mat), filtered_image(new cv::Mat)
{
}

AbstractFilter::AbstractFilter(const AbstractFilter &other)
    : raw_image(other.raw_image), filtered_image(other.filtered_image)
{
}

AbstractFilter &AbstractFilter::operator=(const AbstractFilter &other)
{
    if (this != &other)
    {
        this->raw_image = other.raw_image;
        this->filtered_image = other.filtered_image;
    }

    return *this;
}

AbstractFilter::~AbstractFilter()
{
    delete raw_image;
    delete filtered_image;
}

bool AbstractFilter::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }

    return true;
}

bool AbstractFilter::show()
{
    if (filtered_image->empty())
    {
        std::cerr << "Error: image not filtered" << std::endl;
        return false;
    }

    cv::namedWindow("Filtered", cv::WINDOW_NORMAL);
    cv::resizeWindow("Filtered", 640, 480);
    cv::imshow("Filtered", *filtered_image);


    cv::waitKey(0);
    cv::destroyAllWindows();
    return true;
}

bool AbstractFilter::save(std::string output)
{
    if (filtered_image->empty())
    {
        std::cerr << "Error: image not filtered" << std::endl;
        return false;
    }

    if (cv::imwrite(output, *filtered_image))
    {
        std::cerr << "Error: Failed to save the image" << std::endl;
        return false;
    }

    return true;
}
