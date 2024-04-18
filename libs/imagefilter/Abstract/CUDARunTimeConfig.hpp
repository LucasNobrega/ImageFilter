#ifndef CUDA_RUNTIME_CONFIG_HPP
#define CUDA_RUNTIME_CONFIG_HPP

struct CUDARunTimeConfig
{
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;

    bool automatic_grid_size;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
};

#endif