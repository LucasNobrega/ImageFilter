#ifndef SHARPENING_FILTER_HPP
#define SHARPENING_FILTER_HPP

#include "AbstractFilter.hpp"


class SharpeningFilter : public AbstractFilter
{
public:
    SharpeningFilter();
    SharpeningFilter(const SharpeningFilter &other);
    SharpeningFilter &operator=(const SharpeningFilter &other);
    ~SharpeningFilter();

    bool apply();
};


#endif