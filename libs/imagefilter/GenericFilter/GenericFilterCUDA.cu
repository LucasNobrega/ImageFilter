#include "GenericFilterCUDA.hpp"
#include <stdio.h>

__global__ void idleKernel2(unsigned char *srcImage,
                           unsigned char *dstImage,
                           int filter_width,
                           int filter_height,
                           unsigned int width,
                           unsigned int height,
                           int channel)
{
    printf("idleKernel2\n");
}

__global__ void genericFilterKernel(unsigned char *srcImage,
                                    unsigned char *dstImage,
                                    unsigned int img_width,
                                    unsigned int img_height,
                                    float *kernel,
                                    int kernel_width,
                                    int kernel_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = threadIdx.z; 
    int channels = blockDim.z;

    // only threads inside image will write results
    if ((x >= kernel_width / 2) && (x < (img_width - kernel_width / 2)) &&
        (y >= kernel_height / 2) && (y < (img_height - kernel_height / 2)))
    {
        // Sum of pixel values
        float sum = 0;
        // Number of filter pixels
        float kS = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -kernel_height / 2; ky <= kernel_height / 2; ky++)
        {
            for (int kx = -kernel_width / 2; kx <= kernel_width / 2; kx++)
            {
                float fl = srcImage[((y + ky) * img_width + (x + kx)) * channels + c];
                float kl = kernel[(ky + kernel_height / 2) * kernel_width + (kx + kernel_width / 2)];
                sum += fl * kl;
                kS += kl;
            }
        }
        dstImage[(y * img_width + x) * channels + c] = sum / kS;
    }

}

GenericFilterCUDA::GenericFilterCUDA(CUDARunTimeConfig cuda_run_time_config)
    : AbstractFilterCUDA(1, 1, cuda_run_time_config)
{
    cuda_kernel = &idleKernel2;
    generic_cuda_kernel = &genericFilterKernel;
}

GenericFilterCUDA::GenericFilterCUDA(const GenericFilterCUDA &other)
    : AbstractFilterCUDA(other)
{
    generic_cuda_kernel = other.generic_cuda_kernel;
    kernel = other.kernel;
}

GenericFilterCUDA &GenericFilterCUDA::operator=(const GenericFilterCUDA &other)
{
    if (this != &other)
    {
        AbstractFilterCUDA::operator=(other);
        this->generic_cuda_kernel = other.generic_cuda_kernel;
        this->kernel = other.kernel;
    }

    return *this;
}

GenericFilterCUDA::~GenericFilterCUDA()
{
}

bool GenericFilterCUDA::configureKernel(const std::string& filename) {
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

void GenericFilterCUDA::activate(cv::Mat *input, cv::Mat *output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int channel = input->step / input->cols;

    const int inputSize = input->cols * input->rows * channel;
    const int outputSize = output->cols * output->rows * channel;
    unsigned char *d_input, *d_output;
    float* d_kernel;
    std::cout << "input->rows: " << input->rows << "\n";
    std::cout << "input->cols: " << input->cols << "\n";
    std::cout << "input->channels: " << channel << "\n";
    std::cout << "output->rows: " << output->rows << "\n";
    std::cout << "output->cols: " << output->cols << "\n";
    std::cout << "output->channels: " << channel << "\n";

    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);
    cudaMalloc<float>(&d_kernel, kernel.cols * kernel.rows);
    cudaMemcpy(d_input, input->ptr(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.ptr<float>(), kernel.cols * kernel.rows, cudaMemcpyHostToDevice);

    if (cuda_run_time_config.automatic_grid_size)
    {
        configGrid();
        cuda_run_time_config.block_z = channel;
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
    (*generic_cuda_kernel)<<<grid, block>>>(d_input,
                                            d_output,
                                            output->cols,
                                            output->rows,
                                            d_kernel, 
                                            kernel.cols,
                                            kernel.rows);
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