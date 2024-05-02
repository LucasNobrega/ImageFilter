#include "Benchmarker.hpp"


Benchmarker::Benchmarker(std::vector<AbstractFilter*> f, std::vector<CUDARunTimeConfig> c) 
    : filters(f), cudaConfigs(c)
{
}