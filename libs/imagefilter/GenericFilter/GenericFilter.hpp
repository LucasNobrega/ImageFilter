#ifndef GENERIC_FILTER_HPP
#define GENERIC_FILTER_HPP

#include "AbstractFilter.hpp"
#include <fstream>

class GenericFilter : public AbstractFilter
{
public:
    GenericFilter();
    GenericFilter(const GenericFilter &other);
    GenericFilter &operator=(const GenericFilter &other);
    ~GenericFilter();

    bool configureKernel(const std::string &kernel_path);
    bool apply();

private:
    cv::Mat kernel;
};


#endif