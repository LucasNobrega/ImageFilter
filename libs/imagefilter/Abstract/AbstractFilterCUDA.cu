#include "AbstractFilterCUDA.hpp"
#include <stdio.h>

__global__ void idleKernel(unsigned char *srcImage,
                           unsigned char *dstImage,
                           int filter_width,
                           int filter_height,
                           unsigned int width,
                           unsigned int height,
                           int channel)
{
}

AbstractFilterCUDA::AbstractFilterCUDA(unsigned int filter_kernel_width,
                                       unsigned int filter_kernel_height,
                                       CUDARunTimeConfig cuda_run_time_config)
    : AbstractFilter(), filter_kernel_width(filter_kernel_width),
      filter_kernel_height(filter_kernel_height),
      cuda_run_time_config(cuda_run_time_config), cuda_kernel(idleKernel)
{
}

AbstractFilterCUDA::AbstractFilterCUDA(const AbstractFilterCUDA &other)
    : AbstractFilter(other), cuda_kernel(other.cuda_kernel),
      filter_kernel_width(other.filter_kernel_width),
      filter_kernel_height(other.filter_kernel_height),
      cuda_run_time_config(other.cuda_run_time_config)
{
}

AbstractFilterCUDA &AbstractFilterCUDA::operator=(
    const AbstractFilterCUDA &other)
{
    if (this != &other)
    {
        AbstractFilter::operator=(other);
        this->cuda_kernel = other.cuda_kernel;
        this->filter_kernel_width = other.filter_kernel_width;
        this->filter_kernel_height = other.filter_kernel_height;
        this->cuda_run_time_config = other.cuda_run_time_config;
    }

    return *this;
}

AbstractFilterCUDA::~AbstractFilterCUDA()
{
}

bool AbstractFilterCUDA::read(std::string input)
{
    *raw_image = cv::imread(input);
    if (raw_image->empty())
    {
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

void AbstractFilterCUDA::configGrid()
{
    cuda_run_time_config.grid_x =
        (raw_image->cols + cuda_run_time_config.block_x - 1) /
        cuda_run_time_config.block_x;
    cuda_run_time_config.grid_y =
        (raw_image->rows + cuda_run_time_config.block_y - 1) /
        cuda_run_time_config.block_y;
    cuda_run_time_config.grid_z = 1;
}

bool AbstractFilterCUDA::checkCUDAConfig()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    if (cuda_run_time_config.block_x > props.maxThreadsDim[0] ||
        cuda_run_time_config.block_y > props.maxThreadsDim[1] ||
        cuda_run_time_config.block_z > props.maxThreadsDim[2])
    {
        std::cerr << "Error: Block size exceeds the maximum thread dimension"
                  << std::endl;
        return false;
    }

    if (cuda_run_time_config.block_x == 0 ||
        cuda_run_time_config.block_y == 0 || cuda_run_time_config.block_z == 0)
    {
        std::cerr << "Error: cuda_run_time_config.block size is not set"
                  << std::endl;
        return false;
    }
    if (cuda_run_time_config.block_x > 1024 ||
        cuda_run_time_config.block_y > 1024 ||
        cuda_run_time_config.block_z > 64)
    {
        std::cerr << "Error: cuda_run_time_config.block size is too large"
                  << std::endl;
        return false;
    }
    if (cuda_run_time_config.block_x * cuda_run_time_config.block_y *
            cuda_run_time_config.block_z >
        1024)
    {
        std::cerr << "Error: cuda_run_time_config.block size is too large"
                  << std::endl;
        return false;
    }

    if (cuda_run_time_config.grid_x > props.maxGridSize[0] ||
        cuda_run_time_config.grid_y > props.maxGridSize[1] ||
        cuda_run_time_config.grid_z > props.maxGridSize[2])
    {
        std::cerr << "Error: cuda_run_time_config.grid size exceeds the "
                     "maximum cuda_run_time_config.grid dimension"
                  << std::endl;
        return false;
    }

    if (cuda_run_time_config.grid_x == 0 || cuda_run_time_config.grid_y == 0 ||
        cuda_run_time_config.grid_z == 0)
    {
        std::cerr << "Error: cuda_run_time_config.grid size is not set"
                  << std::endl;
        return false;
    }

    if (cuda_run_time_config.grid_x > 65535 ||
        cuda_run_time_config.grid_y > 65535 ||
        cuda_run_time_config.grid_z > 65535)
    {
        std::cerr << "Error: cuda_run_time_config.grid size is too large"
                  << std::endl;
        return false;
    }

    if (cuda_run_time_config.grid_x * cuda_run_time_config.grid_y *
            cuda_run_time_config.grid_z >
        65535)
    {
        std::cerr << "Error: cuda_run_time_config.grid size is too large"
                  << std::endl;
        return false;
    }

    return true;
}

void AbstractFilterCUDA::activate(cv::Mat *input, cv::Mat *output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input->step / input->cols;

    const int inputSize = input->cols * input->rows * channel;
    const int outputSize = output->cols * output->rows * channel;
    unsigned char *d_input, *d_output;
    std::cout << "input->rows: " << input->rows << "\n";
    std::cout << "input->cols: " << input->cols << "\n";
    std::cout << "output->rows: " << output->rows << "\n";
    std::cout << "output->cols: " << output->cols << "\n";

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);
    cudaMemcpy(d_input, input->ptr(), inputSize, cudaMemcpyHostToDevice);

    if (cuda_run_time_config.automatic_grid_size)
    {
        configGrid();
    }

    if (!checkCUDAConfig())
    {
        return;
    }

    std::cout << "cuda_run_time_config.block_x: "
              << cuda_run_time_config.block_x << "\n"
              << "cuda_run_time_config.block_y: "
              << cuda_run_time_config.block_y << "\n"
              << "cuda_run_time_config.block_z: "
              << cuda_run_time_config.block_z << "\n";

    std::cout << "cuda_run_time_config.grid_x: " << cuda_run_time_config.grid_x
              << "\n"
              << "cuda_run_time_config.grid_y: " << cuda_run_time_config.grid_y
              << "\n"
              << "cuda_run_time_config.grid_z: " << cuda_run_time_config.grid_z
              << "\n";

    cudaEventRecord(start);

    dim3 block(cuda_run_time_config.block_x,
               cuda_run_time_config.block_y,
               cuda_run_time_config.block_z);
    dim3 grid(cuda_run_time_config.grid_x,
              cuda_run_time_config.grid_y,
              cuda_run_time_config.grid_z);
    (*cuda_kernel)<<<grid, block>>>(d_input,
                                    d_output,
                                    filter_kernel_width,
                                    filter_kernel_height,
                                    output->cols,
                                    output->rows,
                                    channel);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(output->ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}