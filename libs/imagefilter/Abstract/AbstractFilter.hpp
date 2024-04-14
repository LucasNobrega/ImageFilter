#ifndef IMAGE_FILTER_LIB_H
#define IMAGE_FILTER_LIB_H

#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
//#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <stdio.h>

class AbstractFilter {
public:
    AbstractFilter();
    AbstractFilter(const AbstractFilter& other);
    AbstractFilter& operator=(const AbstractFilter& other);

    virtual bool read(std::string input);
    virtual bool apply() = 0;
    virtual bool show();
    virtual bool save(std::string output);
    
    virtual ~AbstractFilter();

protected:
    cv::Mat *raw_image;
    cv::Mat *filtered_image;
};

#endif