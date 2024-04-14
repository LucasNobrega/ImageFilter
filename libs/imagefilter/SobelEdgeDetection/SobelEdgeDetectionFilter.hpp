#ifndef SOBELEDGEDETECTION_FILTER_HPP
#define SOBELEDGEDETECTION_FILTER_HPP

#include "AbstractFilter.hpp"


class SobelEdgeDetectionFilter : public AbstractFilter {
public:
    SobelEdgeDetectionFilter();
    SobelEdgeDetectionFilter(const SobelEdgeDetectionFilter& other);
    SobelEdgeDetectionFilter& operator=(const SobelEdgeDetectionFilter& other);
    ~SobelEdgeDetectionFilter();

    bool apply();
    bool read(std::string input) override;

};


#endif