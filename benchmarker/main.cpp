#include "config.hpp"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#include "BoxFilter.hpp"
#include "BoxFilterCUDA.hpp"
#include "LaplacianFilter.hpp"
#include "LaplacianFilterCUDA.hpp"
#include "MedianBlurFilter.hpp"
#include "MedianBlurFilterCUDA.hpp"
#include "SharpeningFilter.hpp"
#include "SharpeningFilterCUDA.hpp"
#include "SobelEdgeDetectionFilter.hpp"
#include "SobelEdgeDetectionFilterCUDA.hpp"
#include "TotalVariationFilter.hpp"
#include "TotalVariationFilterCUDA.hpp"
#include "Benchmarker.hpp"


int main()
{

    std::cout << "Project: " << projectName << std::endl;
    std::cout << "Version: " << projectVersion << std::endl;

    CUDARunTimeConfig c1;
    c1.block_x = 16;
    c1.block_y = 16;
    c1.block_z = 1;
    c1.automatic_grid_size = true;

    CUDARunTimeConfig c2;
    c2.block_x = 8;
    c2.block_y = 8;
    c2.block_z = 1;
    c2.automatic_grid_size = true;

    Benchmarker b({new BoxFilter(),
                   new LaplacianFilter(),
                   new MedianBlurFilter(),
                   new SharpeningFilter(),
                   new SobelEdgeDetectionFilter(),
                   new TotalVariationFilter()},
                   {c1, c2});

    return 0;
}
