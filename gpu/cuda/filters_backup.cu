
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
rgb2gray_kernel(uchar *inputImage, uchar *grayImage, const int width, const int height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	float r = static_cast< float >(inputImage[(y * width) + x]);
	float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
	float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

	float grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

	grayImage[(y * width) + x] = static_cast< uchar >(grayPix);
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
	error = cudaMalloc((void **) &inputImage_device, width * height * 3 * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer inputImage_device (error code %s)\n");
	error = cudaMalloc((void **) &grayImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	allocationTime.stop();
	
	// Copy the input image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy(inputImage_device, inputImage, width * height * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy inputImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	rgb2gray_kernel<<<blocksPerGrid, threadsPerBlock>>>(inputImage_device, grayImage_device, width, height);
	checkError(cudaGetLastError(), "Failed to launch rgb2gray_kernel (error code %s)\n");
	kernelTime.stop();

	// Copy the grayscale image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy(grayImage, grayImage_device, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy grayImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();
	
	// Free the device buffers
	cudaFree(inputImage_device);
	cudaFree(grayImage_device);
	
	// Print the timers
	cout << fixed << setprecision(6);
	cout << "rgb2gray (allocation): \t\t\t" << allocationTime.getElapsed() << " seconds." << endl;
	cout << "rgb2gray (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "rgb2gray (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "rgb2gray (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__global__ void
histogram1D_kernel(uchar *grayImage, const int width, const int height, uint *histogram) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	atomicAdd(&histogram[static_cast< uint >(grayImage[(y * width) + x])], 1);	
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
	error = cudaMalloc((void **) &grayImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	error = cudaMalloc((void **) &histogram_device, HISTOGRAM_SIZE * sizeof(uint));
	checkError(error, "Failed to allocate device buffer histogram_device (error code %s)\n");
	allocationTime.stop();
	
	// Set histogram buffer to 0
	error = cudaMemset(reinterpret_cast< void * >(histogram_device), 0, HISTOGRAM_SIZE * sizeof(uint));
	checkError(error, "Failed to set histogram buffer to 0 (error code %s)\n");
	
	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy(grayImage_device, grayImage, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	histogram1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, histogram_device);
	checkError(cudaGetLastError(), "Failed to launch histogram1D_kernel (error code %s)\n");
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
	cout << "histogram1D (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	cout << "histogram1D (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "histogram1D (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "histogram1D (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__global__ void
contrast1D_kernel(uchar *grayImage, const int width, const int height, uint min, uint max, float diff) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	uchar pixel = grayImage[(y * width) + x];

	if ( pixel < min ) {
		pixel = 0;
	}
	else if ( pixel > max ) {
		pixel = 255;
	}
	else {
		pixel = static_cast< uchar >(255.0f * (pixel - min) / diff);
	}
	
	grayImage[(y * width) + x] = pixel;
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
	error = cudaMalloc((void **) &grayImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	allocationTime.stop();
	
	// 
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
	float diff = max - min;
	
	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy(grayImage_device, grayImage, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	contrast1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, min, max, diff);
	checkError(cudaGetLastError(), "Failed to launch contrast1D_kernel (error code %s)\n");
	kernelTime.stop();
	
	// Copy the grayscale image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy(grayImage, grayImage_device, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy grayImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();
	
	// Free the device buffer
	cudaFree(grayImage_device);
	
	// Print the timers
	cout << fixed << setprecision(6);
	cout << "contrast1D (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	cout << "contrast1D (copyToDevice): \t\t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "contrast1D (kernel): \t\t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "contrast1D (copyFromDevice): \t\t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

__global__ void
triangularSmooth_kernel(uchar *grayImage, uchar *smoothImage, const int width, const int height, const float *filter) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	// Make sure we are within bounds
	if (x >= width || y >= height) return;
	
	uint filterItem = 0;
	float filterSum = 0.0f;
	float smoothPix = 0.0f;

	// unroll?
	for ( int fy = y - 2; fy < y + 3; fy++ ) {
		for ( int fx = x - 2; fx < x + 3; fx++ ) {
			if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) {
				filterItem++;
				continue;
			}

			smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
			filterSum += filter[filterItem];
			filterItem++;
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
	float *filter_device;
	error = cudaMalloc((void **) &grayImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	error = cudaMalloc((void **) &smoothImage_device, width * height * sizeof(uchar));
	checkError(error, "Failed to allocate device buffer smoothImage_device (error code %s)\n");
	error = cudaMalloc((void **) &filter_device, FILTER_SIZE * sizeof(float));
	checkError(error, "Failed to allocate device buffer filter_device (error code %s)\n");
	allocationTime.stop();
	
	// Copy the grayscale image and the filter from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy(grayImage_device, grayImage, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	error = cudaMemcpy(filter_device, filter, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	triangularSmooth_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device, width, height, filter_device);
	checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel (error code %s)\n");
	kernelTime.stop();
	
	// Copy the histogram from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy(smoothImage, smoothImage_device, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	checkError(error, "Failed to copy smoothImage from device to host (error code %s)\n");
	copyFromDeviceTime.stop();
	
	// Free the device buffers
	cudaFree(grayImage_device);
	cudaFree(smoothImage_device);
	cudaFree(filter_device);
	
	// Print the timers
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (copyToDevice): \t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (copyFromDevice): \t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

