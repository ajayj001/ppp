
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

	float r = static_cast< float >(inputImage[(y * pitch) + x]);
	float g = static_cast< float >(inputImage[(pitch * height) + (y * pitch) + x]);
	float b = static_cast< float >(inputImage[(2 * pitch * height) + (y * pitch) + x]);

	float grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

	grayImage[(y * pitch) + x] = static_cast< uchar >(grayPix);
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
	dim3 threadsPerBlock(128, 4);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
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
	const int x_base = (blockDim.x * blockIdx.x + threadIdx.x) * HISTOGRAM_PIXELS_WIDTH;
	const int y_base = (blockDim.y * blockIdx.y + threadIdx.y) * HISTOGRAM_PIXELS_HEIGHT;
	const int histogram_index = blockDim.y * threadIdx.y + threadIdx.x;

	// Initialize shared histogram
	__shared__ uchar histogram_shared[HISTOGRAM_SIZE];
	histogram_shared[histogram_index] = 0;

	for (int x = x_base; x < x_base + HISTOGRAM_PIXELS_WIDTH; x++) {
		for (int y = y_base; y < y_base + HISTOGRAM_PIXELS_HEIGHT; y++) {
			// Make sure we are within bounds
			if (x >= width || y >= height) continue;

			// Add pixel data to shared histogram
			histogram_shared[static_cast< uint >(grayImage[(y * pitch) + x])]++;
		}
	}

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
	dim3 blocksPerGrid(ceil((float)width / HISTOGRAM_PIXELS_WIDTH / threadsPerBlock.x), ceil((float)height / HISTOGRAM_PIXELS_HEIGHT / threadsPerBlock.y));
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

__global__ void
contrast1D_kernel(uchar *grayImage, const int width, const int height, const uint min, const uint max, const float diff, const size_t pitch) {
	const int x_base = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int x = x_base; x < x_base + CONTRAST1D_PIXELS_PER_THREAD; x++) {
		// Make sure we are within bounds
		if (x >= width || y >= height) continue;

		uchar pixel = grayImage[(y * pitch) + x];

		if ( pixel < min ) {
			pixel = 0;
		}
		else if ( pixel > max ) {
			pixel = 255;
		}
		else {
			pixel = static_cast< uchar >(255.0f * (pixel - min) / diff);
		}

		grayImage[(y * pitch) + x] = pixel;
	}
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
	float diff = max - min;

	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar), height, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / CONTRAST1D_PIXELS_PER_THREAD / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	contrast1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, min, max, diff, pitch);
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

__constant__ float filter_constant[FILTER_LENGTH];

__global__ void
triangularSmooth_kernel(uchar *grayImage, uchar *smoothImage, const int width, const int height, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Make sure we are within bounds
	if (x >= width || y >= height) return;

	uint filterItem = 0;
	float filterSum = 0.0f;
	float smoothPix = 0.0f;

	for ( int fy = y - 2; fy < y + 3; fy++ ) {
		for ( int fx = x - 2; fx < x + 3; fx++ ) {
			if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) {
				filterItem++;
				continue;
			}

			smoothPix += grayImage[(fy * pitch) + fx] * filter_constant[filterItem];
			filterSum += filter_constant[filterItem];
			filterItem++;
		}
	}

	smoothPix /= filterSum;
	smoothImage[(y * pitch) + x] = static_cast< uchar >(smoothPix);
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
	size_t pitch;
	error = cudaMallocPitch(&grayImage_device, &pitch, width * sizeof(uchar), height);
	checkError(error, "Failed to allocate device buffer grayImage_device (error code %s)\n");
	error = cudaMallocPitch(&smoothImage_device, &pitch, width * sizeof(uchar), height);
	checkError(error, "Failed to allocate device buffer smoothImage_device (error code %s)\n");
	allocationTime.stop();

	// Copy the grayscale image and the filter from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar), height, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	error = cudaMemcpyToSymbol(filter_constant, filter, FILTER_LENGTH * sizeof(float));
	checkError(error, "Failed to copy filter from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	triangularSmooth_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device, width, height, pitch);
	checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the smooth image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy2D(smoothImage, width, smoothImage_device, pitch, width * sizeof(uchar), height, cudaMemcpyDeviceToHost);
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

