#ifndef BENCHMARKER_HPP
#define BENCHMARKER_HPP

#include "AbstractFilter.hpp"
#include "AbstractFilterCUDA.hpp"

class Benchmarker {
public:
    Benchmarker(std::vector<AbstractFilter*> filters, std::vector<CUDARunTimeConfig> cudaConfigs); 

private:
    std::vector<AbstractFilter*> filters;
    std::vector<CUDARunTimeConfig> cudaConfigs;
};


#endif