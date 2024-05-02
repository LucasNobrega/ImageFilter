#include "GenericFilter.hpp"

GenericFilter::GenericFilter() : AbstractFilter(), kernel(cv::Mat())
{
}

GenericFilter::GenericFilter(const GenericFilter &other)
    : AbstractFilter(other), kernel(other.kernel)
{
}

GenericFilter &GenericFilter::operator=(const GenericFilter &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
        this->kernel= other.kernel;
    }

    return *this;
}

GenericFilter::~GenericFilter()
{
}

bool GenericFilter::configureKernel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<float>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        float val;
        while (iss >> val) {
            row.push_back(val);
        }
        data.push_back(row);
    }

    size_t rows = data.size();
    size_t cols = data[0].size();
    cv::Mat k(int(rows), int(cols), CV_32F);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            k.at<float>(int(i), int(j)) = data[i][j];
        }
    }

    this->kernel = k;
    return true;
}

bool GenericFilter::apply()
{
    if (raw_image->empty())
    {
        std::cerr << "Error: raw image not loaded" << std::endl;
        return false;
    }

    cv::filter2D(*raw_image,
                  *filtered_image,
                  -1,
                  kernel);
    return true;
}
