#include "AbstractFilterCUDA.hpp"
#include <stdio.h>

__global__ void idleKernel(unsigned char *srcImage, unsigned char *dstImage, int filter_width, int filter_height, unsigned int width, unsigned int height, int channel)
{
}

AbstractFilterCUDA::AbstractFilterCUDA( unsigned int filter_kernel_width,
                        unsigned int filter_kernel_height,
                        unsigned int cuda_block_size)
: AbstractFilter()
, filter_kernel_width(filter_kernel_width)
, filter_kernel_height(filter_kernel_height)
, cuda_block_size(cuda_block_size)
, cuda_kernel(idleKernel)
{
}

AbstractFilterCUDA::AbstractFilterCUDA(const AbstractFilterCUDA& other)
: AbstractFilter(other)
, cuda_kernel(other.cuda_kernel)
, filter_kernel_width(other.filter_kernel_width)
, filter_kernel_height(other.filter_kernel_height)
, cuda_block_size(other.cuda_block_size)
{
}

AbstractFilterCUDA& AbstractFilterCUDA::operator=(const AbstractFilterCUDA& other) {
    if(this != &other){
        AbstractFilter::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
        this->filter_kernel_width = other.filter_kernel_width;
        this->filter_kernel_height = other.filter_kernel_height;
        this->cuda_block_size = other.cuda_block_size;
    }

    return *this;
}

AbstractFilterCUDA::~AbstractFilterCUDA(){
}

bool AbstractFilterCUDA::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return false;
    }

    filtered_image->create(raw_image->size(), raw_image->type());

    return true;
}

bool AbstractFilterCUDA::apply()
{
    activate(raw_image, filtered_image);

    return true;
}

void AbstractFilterCUDA::activate(cv::Mat* input, cv::Mat* output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input->step/input->cols; 

    const int inputSize = input->cols * input->rows * channel;
    const int outputSize = output->cols * output->rows * channel;
    unsigned char *d_input, *d_output;
    std::cout << "input->rows: " << input->rows << "\n";
    std::cout << "input->cols: " << input->cols << "\n";
    std::cout << "output->rows: " << output->rows << "\n";
    std::cout << "output->cols: " << output->cols << "\n";
    
    cudaMalloc<unsigned char>(&d_input,inputSize);
    cudaMalloc<unsigned char>(&d_output,outputSize);
    cudaMemcpy(d_input,input->ptr(),inputSize,cudaMemcpyHostToDevice);

    const dim3 block(cuda_block_size,cuda_block_size);
    const dim3 grid((output->cols + block.x - 1)/block.x, (output->rows + block.y - 1)/block.y);
    std::cout << "block: " << block.x << ", " << block.y << ", " << block.z << "\n";
    std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << "\n";

    cudaEventRecord(start);

    (*cuda_kernel)<<<grid,block>>>(d_input, d_output, filter_kernel_width, filter_kernel_height, output->cols, output->rows, channel);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(error));
    }


    cudaMemcpy(output->ptr(),d_output,outputSize,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<< "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}