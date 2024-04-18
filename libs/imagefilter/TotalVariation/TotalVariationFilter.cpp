#include "TotalVariationFilter.hpp"

TotalVariationFilter::TotalVariationFilter() : AbstractFilter()
{
}

TotalVariationFilter::TotalVariationFilter(const TotalVariationFilter &other)
    : AbstractFilter(other)
{
}

TotalVariationFilter &TotalVariationFilter::operator=(
    const TotalVariationFilter &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
    }

    return *this;
}

TotalVariationFilter::~TotalVariationFilter()
{
}

bool TotalVariationFilter::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }
    cv::cvtColor(*raw_image, *raw_image, cv::COLOR_BGR2GRAY);

    filtered_image->create(raw_image->size(), raw_image->type());

    return true;
}

bool TotalVariationFilter::save(std::string output)
{
    if (filtered_image->empty())
    {
        std::cerr << "Error: image not filtered" << std::endl;
        return false;
    }
    filtered_image->convertTo(*filtered_image, CV_32F, 1.0 / 255, 0);
    (*filtered_image) *= 255;

    if (cv::imwrite(output, *filtered_image))
    {
        std::cerr << "Error: Failed to save the image" << std::endl;
        return false;
    }

    return true;
}

bool TotalVariationFilter::apply()
{
    if (raw_image->empty())
    {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::Point anchor = cv::Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int kernel_size = 3;

    cv::Mat outputi;
    cv::Mat kernel[8];

    kernel[0] = (cv::Mat_<double>(kernel_size, kernel_size) << -1,
                 0,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0,
                 0);
    kernel[1] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 -1,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0,
                 0);
    kernel[2] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 -1,
                 0,
                 1,
                 0,
                 0,
                 0,
                 0);
    kernel[3] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 0,
                 -1,
                 1,
                 0,
                 0,
                 0,
                 0);
    kernel[4] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 0,
                 0,
                 1,
                 -1,
                 0,
                 0,
                 0);
    kernel[5] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 -1,
                 0,
                 0);
    kernel[6] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 0,
                 -1,
                 0);
    kernel[7] = (cv::Mat_<double>(kernel_size, kernel_size) << 0,
                 0,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0,
                 -1);

    for (int i = 0; i < 8; i++)
    {
        // Apply 2D filter
        cv::filter2D(*raw_image, outputi, ddepth, kernel[i], anchor, delta);

        std::cout << "filtere_image: " << filtered_image->rows << ", "
                  << filtered_image->cols << std::endl;
        std::cout << "output_i: " << outputi.rows << ", " << outputi.cols
                  << std::endl;

        *filtered_image += outputi;
    }

    return true;
}
