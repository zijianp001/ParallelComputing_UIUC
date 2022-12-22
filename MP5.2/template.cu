// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *A, float *B, int len) {
	int curr = blockIdx.x * blockDim.x + threadIdx.x;
	if(curr < len && blockIdx.x > 0) {
		B[curr] = B[curr] + A[blockIdx.x - 1];
	}
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[SECTION_SIZE];
  int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  if(i < len) {
	  XY[threadIdx.x] = input[i];
  }

  if(i + blockDim.x < len) {
	  XY[blockDim.x + threadIdx.x] = input[i + blockDim.x];
  }

  //Dim red
  for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
	  __syncthreads();
	  int curr = 2 * stride * (threadIdx.x + 1) - 1;
	  if(curr < SECTION_SIZE) {
		  XY[curr] = XY[curr] + XY[curr - stride];
	  }
  }

  for(unsigned int stride = 256; stride > 0; stride /= 2) {
	  __syncthreads();
	  int curr = 2 * stride * (threadIdx.x + 1) -1;
	  if(curr + stride < SECTION_SIZE) {
		  XY[curr + stride] = XY[curr + stride] + XY[curr];
	  }
  }


  __syncthreads();
  if(i < len) {
	  output[i] = XY[threadIdx.x];
  }

  if(i + blockDim.x < len) {
	  output[i + blockDim.x] = XY[blockDim.x + threadIdx.x];
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *hostTemp;
  float *deviceTemp;

  int numElements; // number of elements

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  int num_sec = numElements / SECTION_SIZE;
  hostTemp = (float *)malloc(num_sec * sizeof(float));


  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  cudaMalloc((void **)&deviceTemp, num_sec * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here


  int size_output = numElements * sizeof(float);
  dim3 DimGrid1(ceil(numElements / (SECTION_SIZE * 1.0)), 1, 1);
  dim3 DimBlock1(BLOCK_SIZE, 1, 1); 
  dim3 DimGrid2(ceil(num_sec/ (SECTION_SIZE * 1.0)), 1, 1);
  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  dim3 DimBlock3(SECTION_SIZE, 1, 1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid1, DimBlock1>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutput, deviceOutput, size_output, cudaMemcpyDeviceToHost);

  if(num_sec > 0) {
	  for(int i = 0; i < num_sec; i++) {
		  hostTemp[i] = hostOutput[SECTION_SIZE*(i+1)-1];
	  }
	  cudaMemcpy(deviceTemp, hostTemp, num_sec * sizeof(float), cudaMemcpyHostToDevice);
	  scan<<<DimGrid2, DimBlock2>>>(deviceTemp, deviceTemp, num_sec);
	  cudaDeviceSynchronize();

	  add<<<DimGrid1, DimBlock3>>>(deviceTemp, deviceOutput, numElements);
	  cudaDeviceSynchronize();
	  cudaMemcpy(hostOutput, deviceOutput, size_output, cudaMemcpyDeviceToHost);
  }

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");




  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceTemp);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(hostTemp);

  return 0;
}
