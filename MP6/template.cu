// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128
#define SECTION_SIZE 256

//@@ insert code here

__global__ void imageToFloat(float* inputImage, unsigned char* ucharImage, int width, int height, int channels) {
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii < width * height * channels) {
		ucharImage[ii] = (unsigned char) (255 * inputImage[ii]); 
	}
}

__global__ void RGBToGrayScale(unsigned char* ucharImage, unsigned char* gray, int width, int height) {
	int ii = blockDim.y * blockIdx.y + threadIdx.y;
	int jj = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii < height && jj < width) {
		int idx = ii * width + jj;
		unsigned char r = ucharImage[3 * idx];
		unsigned char g = ucharImage[3 * idx + 1];
		unsigned char b = ucharImage[3 * idx + 2];
		gray[idx] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
	}
}

__global__ void histogramOfGrayImage(unsigned char* gray, unsigned int* histogram, int width, int height) {
	__shared__ unsigned int tile[256];
	if(threadIdx.x < 256) {
		tile[threadIdx.x] = 0;
	}

	__syncthreads();
	int ii = blockDim.x * blockIdx.x + threadIdx.x;
	int size = width * height;
	int stride = blockDim.x * gridDim.x;
	while(ii < size) {
		atomicAdd(&(tile[gray[ii]]), 1);
		ii += stride;
	}

	__syncthreads();
	if(threadIdx.x < 256) {
		atomicAdd(&(histogram[threadIdx.x]), tile[threadIdx.x]);
	}
}

__global__ void scan(unsigned int* input, float* output, int width, int height, int len) {
	__shared__ float XY[SECTION_SIZE];
	int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	if(i < len) {
		XY[threadIdx.x] = input[i];
	}
	if(i + blockDim.x < len) {
		XY[blockDim.x + threadIdx.x] = input[i + blockDim.x];
	}
	
	for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		int curr = 2 * stride * (threadIdx.x + 1) - 1;
		if(curr < SECTION_SIZE) {
			XY[curr] = XY[curr] + XY[curr - stride];
		}
	}
	
	for(unsigned int stride = 64; stride > 0; stride /= 2) {
		__syncthreads();
		int curr = 2 * stride * (threadIdx.x + 1) -1;
		if(curr + stride < SECTION_SIZE) {
			XY[curr + stride] = XY[curr + stride] + XY[curr];
		}
	}
	
	__syncthreads();
	if(i < len) {
		output[i] = XY[threadIdx.x]/((float)(width * height));
	}
	
	if(i + blockDim.x < len) {
		output[i + blockDim.x] = XY[blockDim.x + threadIdx.x]/((float)(width * height));
	}

}


__global__ void computeMinimum(float* input, unsigned char* output, int width, int height) {
	int ii = blockDim.y * blockIdx.y + threadIdx.y;
        int jj = blockDim.x * blockIdx.x + threadIdx.x;
	if(ii < height && jj < width) {
		int idx = blockIdx.z * width * height + ii * width + jj;
		float equal = 255.0 * (input[output[idx]] - input[0]) / (1.0 - input[0]);
		float clam = min(max(equal, 0.0), 255.0);
		output[idx] = (unsigned char) (clam);
	}
	
}


__global__ void floatToImage(unsigned char* input, float* image, int width, int height, int channels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < width * height * channels) {
		image[idx] = (float) (input[idx]/255.0);
	}

}




int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  float *deviceInputImage;
  unsigned char *ucharImage;
  unsigned char *gray;
  unsigned int *histogram;
  float *prob;
  float *output;



  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
  int ucharSize = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);
  int graySize = imageWidth * imageHeight * sizeof(unsigned char);
  int hisSize = 256 * sizeof(unsigned int);
  int probSize= 256 * sizeof(float);

  cudaMalloc((void **)&deviceInputImage, imageSize);
  cudaMalloc((void **)&ucharImage, ucharSize);
  cudaMalloc((void **)&gray, graySize);
  cudaMalloc((void **)&histogram, hisSize);
  cudaMemset(histogram, 0, 256 * sizeof(unsigned char));
  cudaMalloc((void **)&prob, probSize);
  cudaMalloc((void **)&output, imageSize);

  cudaMemcpy(deviceInputImage, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
  int image = imageWidth * imageHeight * imageChannels;
  dim3 DimGrid1 = dim3(ceil(image/32.0), 1, 1);
  dim3 DimBlock1 = dim3(32, 1, 1);
  imageToFloat<<<DimGrid1, DimBlock1>>>(deviceInputImage, ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();


  dim3 DimGrid2 = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dim3 DimBlock2 = dim3(32, 32, 1);
  RGBToGrayScale<<<DimGrid2, DimBlock2>>>(ucharImage, gray, imageWidth, imageHeight);
  cudaDeviceSynchronize();

   
  int imageG = imageWidth * imageHeight;
  dim3 DimGrid3 = dim3(ceil(imageG/256.0), 1, 1);
  dim3 DimBlock3 = dim3(256, 1, 1);
  histogramOfGrayImage<<<DimGrid3, DimBlock3>>>(gray, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  
  dim3 DimGrid4 = dim3(1, 1, 1);
  dim3 DimBlock4 = dim3(128, 1, 1);
  scan<<<DimGrid4, DimBlock4>>>(histogram, prob, imageWidth, imageHeight, 256);
  cudaDeviceSynchronize();

  
  dim3 DimGrid5 = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 DimBlock5 = dim3(32, 32, 1);
  computeMinimum<<<DimGrid5, DimBlock5>>>(prob, ucharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();


  floatToImage<<<DimGrid1, DimBlock1>>>(ucharImage, output, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
    

  cudaMemcpy(hostOutputImageData, output, imageSize, cudaMemcpyDeviceToHost);



  wbSolution(args, outputImage);

  cudaFree(deviceInputImage);
  cudaFree(ucharImage);
  cudaFree(gray);
  cudaFree(histogram);
  cudaFree(prob);
  cudaFree(output);

  //@@ insert code here

  return 0;
}
