#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__constant__ float mask_con[4000];

__global__ void conv_forward_kernel_14(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Width_out, const int Height_out)
{
    


    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define i4(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_con[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    extern __shared__ float N_ds[];
    const int W_size = ceil((Width_out * 1.0) / (TILE_WIDTH * 1.0));
    int m = blockIdx.x;
    int b = blockIdx.z;
    //int h_out = (blockIdx.y / W_size) * TILE_WIDTH + (TILE_WIDTH + K -1);
    //int w_out = (blockIdx.y % W_size) * TILE_WIDTH + (TILE_WIDTH + K -1); 
    int h_start = (blockIdx.y / W_size) * TILE_WIDTH;
    int w_start = (blockIdx.y % W_size) * TILE_WIDTH;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    for(int c = 0; c < Channel; c++) {
	    for(int i = h; i < (blockIdx.y / W_size) * TILE_WIDTH + (TILE_WIDTH + K -1); i+=TILE_WIDTH) {
		    for(int j = w; j < (blockIdx.y % W_size) * TILE_WIDTH + (TILE_WIDTH + K -1); j+=TILE_WIDTH) {
			    if(i < Height && j < Width) {
				    N_ds[(c * (TILE_WIDTH + K -1) * (TILE_WIDTH + K -1)) + (i - h_start) * (TILE_WIDTH + K -1) + (j - w_start)] = i4(b, c, i, j);
			    }
			    else {
				    N_ds[(c * (TILE_WIDTH + K -1) * (TILE_WIDTH + K -1)) + (i - h_start) * (TILE_WIDTH + K -1) + (j - w_start)] = 0.0;
			    }
		    } 
	    }
	    __syncthreads();
	    for(int p = 0; p < K; p++) {
		    for(int q = 0; q < K; q++) {
			    if(threadIdx.y + p < TILE_WIDTH + K -1 && threadIdx.x + q < TILE_WIDTH + K -1){
				    acc += N_ds[(c * (TILE_WIDTH + K -1) * (TILE_WIDTH + K -1)) + (threadIdx.y + p) * (TILE_WIDTH + K -1) + threadIdx.x + q] * mask_4d(m, c, p, q);
			    }
		    }
	    }
	    __syncthreads();

    }
    if(h < Height_out && w < Width_out) {
	    out_4d(b, m, h, w) = acc;
    }



    

    #undef out_4d
    #undef i4
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int input_size = Batch * Channel * Width * Height * sizeof(float);
    int output_size = Batch * Map_out * Width_out * Height_out * sizeof(float);
    int mask_size = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_output_ptr, output_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_con, host_mask, mask_size);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int Width_cnt = ceil((Width_out * 1.0) / (TILE_WIDTH * 1.0));
    int Height_cnt = ceil((Height_out * 1.0) / (TILE_WIDTH * 1.0));
    dim3 DimGrid(Map_out, Width_cnt * Height_cnt, Batch);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel_14<<<DimGrid, DimBlock, Channel * (TILE_WIDTH + K -1) * (TILE_WIDTH + K -1) * sizeof(float)>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K, Width_out, Height_out);


}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    int output_size = Batch * Map_out * Width_out * Height_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
