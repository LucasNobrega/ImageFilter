#ifndef TOTALVARIATION_FILTER_HPP
#define TOTALVARIATION_FILTER_HPP

#include "AbstractFilter.hpp"


class TotalVariationFilter : public AbstractFilter
{
public:
    TotalVariationFilter();
    TotalVariationFilter(const TotalVariationFilter &other);
    TotalVariationFilter &operator=(const TotalVariationFilter &other);
    ~TotalVariationFilter();

    bool read(std::string input) override;
    bool save(std::string output) override;
    bool apply();
};


#endif