
#include "SobelEdgeDetectionFilter.hpp"

SobelEdgeDetectionFilter::SobelEdgeDetectionFilter()
: AbstractFilter()
{
}

SobelEdgeDetectionFilter::SobelEdgeDetectionFilter(const SobelEdgeDetectionFilter& other)
: AbstractFilter(other)
{
}

SobelEdgeDetectionFilter& SobelEdgeDetectionFilter::operator=(const SobelEdgeDetectionFilter& other) {
    if(this != &other){
        AbstractFilter::operator=(other);
    }

    return *this;
}

SobelEdgeDetectionFilter::~SobelEdgeDetectionFilter(){
}

bool SobelEdgeDetectionFilter::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }
    // convert RGB to gray scale
    cv::cvtColor(*raw_image, *raw_image, cv::COLOR_BGR2GRAY);

    return true;
}

bool SobelEdgeDetectionFilter::apply()
{
    if (raw_image->empty()) {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::Point anchor = cv::Point( -1, -1 );
    double delta = 0;
    int ddepth = -1;
    int kernel_size = 3; 

    cv::Mat output1;
    cv::Mat kernel1 = (cv::Mat_<double>(kernel_size,kernel_size) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    /// Apply 2D filter
    cv::filter2D(*raw_image , output1, ddepth, kernel1, anchor, delta);


    cv::Mat output2;
    cv::Mat kernel2 = (cv::Mat_<double>(kernel_size,kernel_size) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    /// Apply 2D filter
    cv::filter2D(*raw_image, output2, ddepth, kernel2, anchor, delta);

    *filtered_image = output1 + output2;

    filtered_image->convertTo(*filtered_image, CV_32F, 1.0 / 255, 0);
    (*filtered_image)*=255;

    return true;
}

