#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define ONE_BATCH 25
#define BLOCK_SIZE 1024




__global__ void unroll_and_mul_10(const float* input, float* output, const float* mask, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int Height_out, const int Width_out, const int aCol, const int bCol) {

#define o4(i3, i2, i1, i0) output[(i3) * (Map_out * bCol) + (i2) * bCol + (i1) * Width_out + i0]
#define i4(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * aCol + (i2) * (K * K) + (i1) * (K) + i0]
	__shared__ half2 Tile_mask[TILE_WIDTH][TILE_WIDTH];
	__shared__ half2 Tile_input[TILE_WIDTH][TILE_WIDTH];
        
        int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int cnt = ceil((float)aCol / TILE_WIDTH);
	const float temp = 0.0;
	half2 Pvalue = __float2half2_rn(temp);

	int c, h, w, p, q, col, row;
        for(int i = 0; i < cnt; ++i) {
		col = i * TILE_WIDTH + threadIdx.x;
		row = i * TILE_WIDTH + threadIdx.y;
		c = col / (K * K);
		h = (col % (K * K)) / K;
		w = (col % (K * K)) % K;

		if(col < aCol && Row < Map_out) {
			Tile_mask[threadIdx.y][threadIdx.x] = __float2half2_rn(mask_4d(Row, c, h, w));
		}
		else {
			Tile_mask[threadIdx.y][threadIdx.x] = __float2half2_rn(temp);
		}

		c = row / (K * K);
		h = Col / (Width - K + 1);
		w = Col % (Width - K + 1);

		p = row % (K * K) / K;
		q = (row % (K * K)) % K;

		if(row < aCol && Col < bCol){
			Tile_input[threadIdx.y][threadIdx.x] = __float2half2_rn(i4(blockIdx.z, c, h+p, w+q));
		}
		else {
			Tile_input[threadIdx.y][threadIdx.x] = __float2half2_rn(temp);
		}

		__syncthreads();

		if(Row < Map_out && Col < bCol){
			for(int j = 0; j < TILE_WIDTH; j++) {
				Pvalue = __hadd2(Pvalue,__hmul2(Tile_mask[threadIdx.y][j], Tile_input[j][threadIdx.x])); 
				//Pvalue = __hcmadd(Tile_mask[threadIdx.y][j], Tile_input[j][threadIdx.x], Pvalue);
			}
		}
		__syncthreads();
	}
	if(Row < Map_out && Col < bCol) {
		const __half2 t = Pvalue;
		o4(blockIdx.z, Row, Col / Width_out, Col % Width_out) = __low2float(t);
	}
	
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

    cudaMalloc((void **)device_input_ptr, Batch * Channel * Width * Height * sizeof(float));
    cudaMalloc((void **)device_output_ptr, Batch * Map_out * (Width - K + 1) * (Height - K + 1) * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Width * Height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
	

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    //const int unroll_Height = Channel * K * K;
    //const int unroll_Width = Height_out * Width_out;

    //float * unrolled;
    //cudaMalloc((void **)&unrolled, ONE_BATCH * (Channel * K * K) * (Height - K + 1) * (Width - K + 1) * sizeof(float));


    const int aCol = Channel * K * K;
    //const int bRow = Channel * K * K;
    const int bCol = (Height - K + 1) * (Width - K + 1);
    //const int cCol = bCol;


    dim3 DimGrid(ceil((1.0 * Height_out * Width_out) / TILE_WIDTH), ceil((1.0 * Map_out)/TILE_WIDTH), Batch);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);


    unroll_and_mul_10<<<DimGrid, DimBlock>>>(device_input, device_output, device_mask, Map_out, Channel, Height, Width, K, Height_out, Width_out, aCol, bCol);


    //cudaFree(unrolled);



}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host

    cudaMemcpy(host_output, device_output, Batch * Map_out * (Width - K + 1) * (Height - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);


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
