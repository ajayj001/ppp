
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include "defines.h"

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;


void checkError(cudaError_t error, const char* description) {
	if (error != cudaSuccess) {
		fprintf(stderr, description, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

__global__ void
rgb2gray_kernel(uchar *inputImage, uchar *grayImage, const int width, const int height, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	// Fetch 3 times 4 pixels from device buffer
	const int quarterPitch = pitch / 4;
	const uchar4 red = ((uchar4*)inputImage)[(y * quarterPitch) + x];
	const uchar4 green = ((uchar4*)inputImage)[(quarterPitch * height) + (y * quarterPitch) + x];
	const uchar4 blue = ((uchar4*)inputImage)[(2 * quarterPitch * height) + (y * quarterPitch) + x];

	// Calculate grey values for 4 pixels
	uchar4 grey;
	grey.x = (RED_COEFFICIENT * red.x) + (GREEN_COEFFICIENT * green.x) + (BLUE_COEFFICIENT * blue.x);
	grey.y = (RED_COEFFICIENT * red.y) + (GREEN_COEFFICIENT * green.y) + (BLUE_COEFFICIENT * blue.y);
	grey.z = (RED_COEFFICIENT * red.z) + (GREEN_COEFFICIENT * green.z) + (BLUE_COEFFICIENT * blue.z);
	grey.w = (RED_COEFFICIENT * red.w) + (GREEN_COEFFICIENT * green.w) + (BLUE_COEFFICIENT * blue.w);

	// Store 4 pixels back to the device buffer
	((uchar4*)grayImage)[(y * quarterPitch) + x] = grey;
}

void rgb2gray(uchar *inputImage, uchar *grayImage, const int width, const int height, NSTimer &timer) {
	cudaError_t error = cudaSuccess;

	// Force the initialization of the device context to make sure the timers are accurate
	error = cudaFree(0);
	checkError(error, "Unable to initialize device context (error code %s)\n");

	// Initialize timers
	NSTimer allocationTime = NSTimer("allocateTime", false, false);
	NSTimer copyToDeviceTime = NSTimer("copyToDeviceTime", false, false);
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer copyFromDeviceTime = NSTimer("copyFromDeviceTime", false, false);

	// Allocate two device buffers
	allocationTime.start();
	uchar *inputImage_device, *grayImage_device;
	size_t pitch;
	error = cudaMallocPitch(&inputImage_device, &pitch, width * sizeof(uchar), height * 3);
	checkError(error, "Failed to allocate device buffer inputImage_device (error code %s)\n");
	error = cudaMallocPitch(&grayImage_device, &pitch, width * sizeof(uchar), height);
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	allocationTime.stop();

	// Copy the input image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(inputImage_device, pitch, inputImage, width * sizeof(uchar), width * sizeof(uchar), height * 3, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy inputImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / 4 / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	rgb2gray_kernel<<<blocksPerGrid, threadsPerBlock>>>(inputImage_device, grayImage_device, width, height, pitch);
	checkError(cudaGetLastError(), "Failed to launch rgb2gray_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the grayscale image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy2D(grayImage, width, grayImage_device, pitch, width * sizeof(uchar), height, cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy grayImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();

	// Free the device buffers
	cudaFree(inputImage_device);
	cudaFree(grayImage_device);

	// Print the timers
	cout << fixed << setprecision(6);
	//cout << "rgb2gray (allocation): \t\t\t" << allocationTime.getElapsed() << " seconds." << endl;
	//cout << "rgb2gray (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "rgb2gray (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	//cout << "rgb2gray (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__global__ void
histogram1D_kernel(uchar *grayImage, const int width, const int height, uint *histogram, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fetch 4 bytes from device buffer
	const uchar4 in = ((uchar4*)grayImage)[(y * pitch / 4) + x];

	// Initialize shared histogram
	const int histogram_index = blockDim.y * threadIdx.y + threadIdx.x;
	__shared__ uchar histogram_shared[HISTOGRAM_SIZE];
	histogram_shared[histogram_index] = 0;
	__syncthreads();

	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	// Add pixel data to shared histogram
	histogram_shared[in.x]++;
	histogram_shared[in.y]++;
	histogram_shared[in.z]++;
	histogram_shared[in.w]++;

	// Atomically add shared histogram to global histogram
	__syncthreads();
	atomicAdd(&histogram[histogram_index], histogram_shared[histogram_index]);
}

void histogram1D(uchar *grayImage, uchar *histogramImage, const int width, const int height, uint *histogram, NSTimer &timer) {
	cudaError_t error = cudaSuccess;

	// Initialize timers
	NSTimer allocationTime = NSTimer("allocateTime", false, false);
	NSTimer copyToDeviceTime = NSTimer("copyToDeviceTime", false, false);
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer copyFromDeviceTime = NSTimer("copyFromDeviceTime", false, false);

	// Allocate two device buffers
	allocationTime.start();
	uchar *grayImage_device;
	uint *histogram_device;
	size_t pitch;
	error = cudaMallocPitch(&grayImage_device, &pitch, width * sizeof(uchar), height);
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	error = cudaMalloc(&histogram_device, HISTOGRAM_SIZE * sizeof(uint));
	checkError(error, "Failed to allocate device buffer histogram_device (error code %s)\n");
	allocationTime.stop();

	// Set histogram buffer to 0
	error = cudaMemset(reinterpret_cast< void * >(histogram_device), 0, HISTOGRAM_SIZE * sizeof(uint));
	checkError(error, "Failed to set histogram buffer to 0 (error code %s)\n");

	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar), height, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16); // Product must be 256
	dim3 blocksPerGrid(ceil((float)width / 4 / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	histogram1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, histogram_device, pitch);
	checkError(cudaGetLastError(), "Failed to launch histogram1D_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the histogram from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy(histogram, histogram_device, HISTOGRAM_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy histogram from device to host (error code %s)\n");
	copyFromDeviceTime.stop();

	// Free the device buffers
	cudaFree(grayImage_device);
	cudaFree(histogram_device);

	// Find maximum in histogram
	uint max = 0;
	for ( uint i = 0; i < HISTOGRAM_SIZE; i++ ) {
		if ( histogram[i] > max ) {
			max = histogram[i];
		}
	}

	// Generate histogram image
	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) {
		uint value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( uint y = 0; y < value; y++ ) {
			for ( uint i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( uint y = value; y < HISTOGRAM_SIZE; y++ ) {
			for ( uint i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	// Print the timers
	cout << fixed << setprecision(6);
	//cout << "histogram1D (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	//cout << "histogram1D (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "histogram1D (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	//cout << "histogram1D (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__device__ void
contrast1D_pixelValue (uchar &pixel, const uint min, const uint diff) {
	float temp = 255.0f * (pixel - min) / diff;
	temp = fminf(temp, 255.0f);
	temp = fmaxf(temp, 0.0f);
	pixel = static_cast< uchar >(temp);
}

__global__ void
contrast1D_kernel(uchar *grayImage, const int width, const int height, const uint min, const uint diff, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fetch 4 pixels from device buffer
	uchar4 pixels = ((uchar4*)grayImage)[(y * pitch / 4) + x];

	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	contrast1D_pixelValue(pixels.x, min, diff);
	contrast1D_pixelValue(pixels.y, min, diff);
	contrast1D_pixelValue(pixels.z, min, diff);
	contrast1D_pixelValue(pixels.w, min, diff);

	// Store 4 pixels back to the device buffer
	((uchar4*)grayImage)[(y * pitch / 4) + x] = pixels;
}

void contrast1D(uchar *grayImage, const int width, const int height, uint *histogram, NSTimer &timer) {
	cudaError_t error = cudaSuccess;

	// Initialize timers
	NSTimer allocationTime = NSTimer("allocateTime", false, false);
	NSTimer copyToDeviceTime = NSTimer("copyToDeviceTime", false, false);
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer copyFromDeviceTime = NSTimer("copyFromDeviceTime", false, false);

	// Allocate device buffer
	allocationTime.start();
	uchar *grayImage_device;
	size_t pitch;
	error = cudaMallocPitch(&grayImage_device, &pitch, width * sizeof(uchar), height);
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	allocationTime.stop();

	// Determine minimum, maximum and their difference of histogram
	uint i = 0;

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i++;
	}
	uint min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i--;
	}
	uint max = i;
	uint diff = max - min;

	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar), height, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / 4 / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	contrast1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, min, diff, pitch);
	checkError(cudaGetLastError(), "Failed to launch contrast1D_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the grayscale image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy2D(grayImage, width, grayImage_device, pitch, width * sizeof(uchar), height, cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy grayImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();

	// Free the device buffer
	cudaFree(grayImage_device);

	// Print the timers
	cout << fixed << setprecision(6);
	//cout << "contrast1D (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	//cout << "contrast1D (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "contrast1D (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	//cout << "contrast1D (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__constant__ float filter_constant[FILTER_SIZE][FILTER_SIZE];

__global__ void
triangularSmooth_kernel(uchar *grayImage, uchar *smoothImage, const int width, const int height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x + 2;
	const int y = blockDim.y * blockIdx.y + threadIdx.y + 2;

	__shared__ uchar grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (SMOOTH_BLOCK_HEIGHT + 4)];

	grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2) + (threadIdx.x + 2)] = grayImage[y * width + x];

	if (threadIdx.x == 0) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2) + (threadIdx.x + 0)] = grayImage[(y + 0) * width + (x - 2)];
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2) + (threadIdx.x + 1)] = grayImage[(y + 0) * width + (x - 1)];
	}

	if (threadIdx.y == 0) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 0) + (threadIdx.x + 2)] = grayImage[(y - 2) * width + (x + 0)];
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 1) + (threadIdx.x + 2)] = grayImage[(y - 1) * width + (x + 0)];
	}

	if (threadIdx.x == SMOOTH_BLOCK_WIDTH - 1) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2) + (threadIdx.x + 3)] = x + 1 < width ? grayImage[(y + 0) * width + (x + 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2) + (threadIdx.x + 4)] = x + 2 < width ? grayImage[(y + 0) * width + (x + 2)] : 0;
	}

	if (threadIdx.y == SMOOTH_BLOCK_HEIGHT - 1) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 3) + (threadIdx.x + 2)] = y + 1 < height ? grayImage[(y + 1) * width + (x + 0)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 4) + (threadIdx.x + 2)] = y + 2 < height ? grayImage[(y + 2) * width + (x + 0)] : 0;
	}

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 0) + (threadIdx.x + 0)] = grayImage[(y - 2) * width + (x - 2)];
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 0) + (threadIdx.x + 1)] = grayImage[(y - 2) * width + (x - 1)];
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 1) + (threadIdx.x + 0)] = grayImage[(y - 1) * width + (x - 2)];
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 1) + (threadIdx.x + 1)] = grayImage[(y - 1) * width + (x - 1)];
	}

	if (threadIdx.x == SMOOTH_BLOCK_WIDTH - 1 && threadIdx.y == 0) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 0) + (threadIdx.x + 3)] = x + 1 < width ? grayImage[(y - 2) * width + (x + 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 0) + (threadIdx.x + 4)] = x + 2 < width ? grayImage[(y - 2) * width + (x + 2)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 1) + (threadIdx.x + 3)] = x + 1 < width ? grayImage[(y - 1) * width + (x + 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 1) + (threadIdx.x + 4)] = x + 2 < width ? grayImage[(y - 1) * width + (x + 2)] : 0;
	}

	if (threadIdx.x == 0 && threadIdx.y == SMOOTH_BLOCK_HEIGHT - 1) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 3) + (threadIdx.x + 0)] = y + 1 < height ? grayImage[(y + 1) * width + (x - 2)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 3) + (threadIdx.x + 1)] = y + 1 < height ? grayImage[(y + 1) * width + (x - 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 4) + (threadIdx.x + 0)] = y + 2 < height ? grayImage[(y + 2) * width + (x - 2)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 4) + (threadIdx.x + 1)] = y + 2 < height ? grayImage[(y + 2) * width + (x - 1)] : 0;
	}

	if (threadIdx.x == SMOOTH_BLOCK_WIDTH - 1 && threadIdx.y == SMOOTH_BLOCK_HEIGHT - 1) {
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 3) + (threadIdx.x + 3)] = y + 1 < height && x + 1 < width ? grayImage[(y + 1) * width + (x + 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 3) + (threadIdx.x + 4)] = y + 1 < height && x + 2 < width ? grayImage[(y + 1) * width + (x + 2)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 4) + (threadIdx.x + 3)] = y + 2 < height && x + 1 < width ? grayImage[(y + 2) * width + (x + 1)] : 0;
		grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 4) + (threadIdx.x + 4)] = y + 2 < height && x + 2 < width ? grayImage[(y + 2) * width + (x + 2)] : 0;
	}
	
	__syncthreads();

	// Make sure we are within bounds
	if (x >= width - 2 || y >= height - 2) return;

	float filterSum = 0.0f;
	float smoothPix = 0.0f;

	for ( int dy = -2; dy <= 2; dy++ ) {
		for ( int dx = -2; dx <= 2; dx++ ) {
			smoothPix += grayImage_shared[(threadIdx.y + dy + 2) * (SMOOTH_BLOCK_WIDTH + 4) + (threadIdx.x + dx + 2)] * filter_constant[dy+2][dx+2];
			filterSum += filter_constant[dy+2][dx+2];
		}
	}

	smoothPix /= filterSum;
	smoothImage[(y * width) + x] = static_cast< uchar >(smoothPix);
}

void triangularSmooth(uchar *grayImage, uchar *smoothImage, const int width, const int height, const float *filter, NSTimer &timer) {
	cudaError_t error = cudaSuccess;

	// Initialize timers
	NSTimer allocationTime = NSTimer("allocateTime", false, false);
	NSTimer copyToDeviceTime = NSTimer("copyToDeviceTime", false, false);
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	NSTimer copyFromDeviceTime = NSTimer("copyFromDeviceTime", false, false);

	// Allocate three device buffers
	allocationTime.start();
	uchar *grayImage_device, *smoothImage_device;
	error = cudaMalloc(&grayImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	error = cudaMalloc(&smoothImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer smoothImage_device (error code %s)\n");
	allocationTime.stop();

	// Copy the grayscale image and the filter from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy(grayImage_device, grayImage, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	error = cudaMemcpyToSymbol(filter_constant, filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
	checkError(error, "Failed to copy filter from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(SMOOTH_BLOCK_WIDTH, SMOOTH_BLOCK_HEIGHT);
	dim3 blocksPerGrid(ceil((float)(width - 4) / threadsPerBlock.x), ceil((float)(height - 4) / threadsPerBlock.y));
	triangularSmooth_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device, width, height);
	checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the smooth image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy(smoothImage, smoothImage_device, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy smoothImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();

	// Free the device buffers
	cudaFree(grayImage_device);
	cudaFree(smoothImage_device);

	// Print the timers
	cout << fixed << setprecision(6);
	//cout << "triangularSmooth (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	//cout << "triangularSmooth (copyToDevice): \t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	//cout << "triangularSmooth (copyFromDevice): \t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

