// MP 5.1 Reduction
// Given a list of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len)
{    
    __shared__ float partialSum[2*BLOCK_SIZE];
  
    // Load data into shared memory. 
    if((blockIdx.x * blockDim.x * 2 + threadIdx.x) < len) {
      partialSum[threadIdx.x] = input[blockIdx.x * blockDim.x * 2 + threadIdx.x];
    }
    else {
      partialSum[threadIdx.x] = 0;
    }
    if((blockIdx.x * blockDim.x * 2 + blockDim.x + threadIdx.x) < len) {
      partialSum[blockDim.x + threadIdx.x] = input[blockIdx.x * blockDim.x * 2 + blockDim.x + threadIdx.x];
    }
    else{
      partialSum[blockDim.x + threadIdx.x] = 0;
    }
  
    // Calculate sum of current block.
    for(int stride=blockDim.x; stride>=1; stride/=2){
      __syncthreads();
      if(blockDim.x + threadIdx.x <= stride) {
        partialSum[blockDim.x + threadIdx.x] += partialSum[blockDim.x + threadIdx.x - stride];
      }
    }
  
    __syncthreads();
  
    // Only one thread in each block writes the result of the current block sum to output.
    if(threadIdx.x == blockDim.x + 1) {
       output[blockIdx.x] = partialSum[blockDim.x + threadIdx.x -1];
    }
}
int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = (numInputElements - 1) / (BLOCK_SIZE << 1) + 1;
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int sizeInput = sizeof(float) * numInputElements;
  int sizeOutput = sizeof(float) * numOutputElements;
  cudaMalloc((void **) &deviceInput, sizeInput);
  cudaMalloc((void **) &deviceOutput, sizeOutput);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceInput, hostInput, sizeInput, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numOutputElements, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /***********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab!
   ***********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
