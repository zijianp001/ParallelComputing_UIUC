#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
__constant__ float mem_Kernel[3][3][3];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
	__shared__ float tile[6][6][6];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row_o = blockIdx.y * 4 + ty;
	int col_o = blockIdx.x * 4 + tx;
	int hei_o = blockIdx.z * 4 + tz;
	int row_i = row_o - (3/2);
	int col_i = col_o - (3/2);
	int hei_i = hei_o - (3/2);
	if((row_i >= 0) && (row_i < y_size) &&
	   (col_i >= 0) && (col_i < x_size) &&
	   (hei_i >= 0) && (hei_i < z_size)){
		tile[ty][tx][tz] = input[hei_i*x_size*y_size + row_i *x_size +
			                 col_i];
	}
	else{
		tile[ty][tx][tz] = 0.0f;
	}
	__syncthreads ();
	if(ty < 4 && tx < 4 && tz < 4){
		float Pvalue = 0.0;
		for(int i=0; i<3; i++){
			for(int j=0; j<3; j++){
				for(int k=0; k<3; k++){
					Pvalue += mem_Kernel[i][j][k] * 
						  tile[i+ty][j+tx][k+tz];
				}
			}
		}
		if(row_o < y_size && col_o < x_size && hei_o < z_size){
			output[hei_o*x_size*y_size + row_o*x_size + col_o]=Pvalue;

		}
	}
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  int size = x_size * y_size * z_size * sizeof(float);
  cudaMalloc((void **) &deviceInput, size);
  cudaMalloc((void **) &deviceOutput, size); 
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, hostInput+3, size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mem_Kernel, hostKernel, kernelLength * sizeof(float),
		 0, cudaMemcpyHostToDevice);


  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*4)), ceil(y_size/(1.0*4)), ceil(z_size/(1.0*4)));
  dim3 dimBlock(6, 6, 6);
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);


  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput+3, deviceOutput, size, cudaMemcpyDeviceToHost);
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
