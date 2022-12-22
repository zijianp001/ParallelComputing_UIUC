#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define ONE_BATCH 25
#define BLOCK_SIZE 1024


__global__ void matrixMultiply(const float *A, const float *B, float *C, const int numAColumns, const int numCRows, const int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
        __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
        __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  

        int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        float Pvalue = 0;
	int cnt = ceil((float)numAColumns / TILE_WIDTH);

        

        for(int q = 0; q < cnt; ++q) {
                if((Row < numCRows) && ((q * TILE_WIDTH + threadIdx.x) < numAColumns)){
                        subTileA[threadIdx.y][threadIdx.x] = A[Row * numAColumns + q * TILE_WIDTH + threadIdx.x];
                }
                else{
                        subTileA[threadIdx.y][threadIdx.x] = 0.0;
                }

                if((Col < numCColumns) && (q * TILE_WIDTH + threadIdx.y) < numAColumns){
                        subTileB[threadIdx.y][threadIdx.x] = B[blockIdx.z * numAColumns * numCColumns + (q * TILE_WIDTH + threadIdx.y) * numCColumns + Col];
                }
                else{
                        subTileB[threadIdx.y][threadIdx.x] = 0.0;
                }
                __syncthreads();

                if(Row < numCRows && Col < numCColumns) {
			for(int k = 0; k < TILE_WIDTH; ++k){
				Pvalue += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
			}
		}
                __syncthreads();
        }
        if(Row < numCRows && Col < numCColumns){
                C[blockIdx.z * numCRows * numCColumns + Row * numCColumns + Col] = Pvalue;
        }

}


__global__ void unroll(const float *input, float *output, const int Channel, const int K, const int Height, const int Width, const int Height_out, const int Width_out, const int unroll_Height, const int unroll_Width) {

    int c, s, h_out, w_out, start_outrow, curr_outrow, p, q;
    int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int by = blockIdx.y;
#define i4(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define o3(i2, i1, i0) output[(i2) * (unroll_Height * unroll_Width) + (i1) * (unroll_Width) + i0] 
    if(t < Channel * unroll_Width) {
	    c = t / unroll_Width;
	    s = t % unroll_Width;

	    h_out = s / Width_out;
	    w_out = s % Width_out;

	    start_outrow = c * K * K;

	    for(p = 0; p < K; p++) {
		    for(q = 0; q < K; q++) {
			    curr_outrow = start_outrow + p * K + q;
			    o3(by, curr_outrow, s) = i4(by, c, h_out + p, w_out + q);
		    }
	    }
    }
#undef i4
#undef o3



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

    float * unrolled;
    cudaMalloc((void **)&unrolled, ONE_BATCH * (Channel * K * K) * (Height - K + 1) * (Width - K + 1) * sizeof(float));


    //const int aCol = Channel * K * K;
    //const int bRow = Channel * K * K;
    //const int bCol = Height_out * Width_out;
    //const int cCol = bCol;



    int size = ceil(((Channel * Height_out * Width_out) * (1.0)) / (BLOCK_SIZE * 1.0)); 


    dim3 DimGrid1(size, ONE_BATCH, 1);
    dim3 DimBlock1(BLOCK_SIZE, 1, 1);

    dim3 DimGrid2(ceil((1.0 * (Height_out * Width_out)) / TILE_WIDTH), ceil((1.0 * Map_out) / TILE_WIDTH), ONE_BATCH);
    dim3 DimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    for(int i = 0; i < Batch; i += 25) {
	    unroll<<<DimGrid1, DimBlock1>>>(device_input + i * Channel * Height * Width, unrolled, Channel, K, Height, Width, Height_out, Width_out, Channel * K * K, Height_out * Width_out);
	    matrixMultiply<<<DimGrid2, DimBlock2>>>(device_mask, unrolled, device_output + i * Map_out * Height_out * Width_out, Channel * K * K, Map_out, Height_out * Width_out);

    }

    cudaFree(unrolled);



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
