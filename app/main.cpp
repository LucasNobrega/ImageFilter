#include "config.hpp"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <CLI/CLI.hpp>

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


int main(int argc, const char **argv)
{
    CLI::App image_filter{"Command line parser"};
    int filter = 6; // Default value
    bool use_cuda = false; // Default value
    image_filter.add_option("-f,--filter", filter, "Filter");
    image_filter.add_flag("-c, --cuda", use_cuda, "Use CUDA");
    CLI11_PARSE(image_filter, argc, argv);

    std::cout << projectName << std::endl;
    std::cout << projectVersion << std::endl;
    std::cout << "Filter value: " << filter << std::endl;
    std::cout << "use cuda: " << use_cuda << std::endl;


    AbstractFilter *img_filter; 
    if(!use_cuda){
        switch (filter) {
            case 1:
                img_filter = new BoxFilter(); 
                break;
            case 2:
                img_filter = new LaplacianFilter(); 
                break;
            case 3:
                img_filter = new MedianBlurFilter(); 
                break;
            case 4:
                img_filter = new SharpeningFilter(); 
                break;
            case 5:
                img_filter = new SobelEdgeDetectionFilter(); 
                break;
            case 6:
                img_filter = new TotalVariationFilter(); 
                break;
            default:
                img_filter = new BoxFilter(); 
                break;
        }
    } else{
        switch (filter) {
            case 1:
                img_filter = new BoxFilterCUDA(3, 3, 16); 
                break;
            case 2:
                img_filter = new LaplacianFilterCUDA(3, 3, 0, 5, 1, 0, 16); 
                break;
            case 3:
                img_filter = new MedianBlurFilterCUDA(11, 11, 16); 
                break;
            case 4:
                img_filter = new SharpeningFilterCUDA(3, 3, 16); 
                break;
            case 5:
                img_filter = new SobelEdgeDetectionFilterCUDA(3, 3, 16); 
                break;
            case 6:
                img_filter = new TotalVariationFilterCUDA(3, 3, 16); 
                break;
            default:
                img_filter = new AbstractFilterCUDA(3, 3, 16); 
                break;
        }
    }

    img_filter->read("../../samples/madmen.jpg");
    img_filter->apply();
    img_filter->save("../../samples/madmen_filtered.jpg");
    img_filter->show();


    return 0;
}
