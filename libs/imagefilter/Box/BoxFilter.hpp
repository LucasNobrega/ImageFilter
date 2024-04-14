#ifndef BOX_FILTER_HPP
#define BOX_FILTER_HPP

#include "AbstractFilter.hpp"


class BoxFilter : public AbstractFilter {
public:
    BoxFilter();
    BoxFilter(const BoxFilter& other);
    BoxFilter& operator=(const BoxFilter& other);
    ~BoxFilter();

    bool apply();

private:
    int kernel_size;
};


#endif