
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

/**
 * Checks whether the last operation completed successfully and prints an error otherwise.
 */
void checkError(cudaError_t error, const char* description) {
	if (error != cudaSuccess) {
		fprintf(stderr, description, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

/**
 * Kernel which converts an RGB image to a gray image. Uses the following optimizations:
 *  - Pixels are fetched in blocks of 4 to reduce the number of memory accesses
 *  - Input and output buffers are pitched to improve coalescing of the memory accesses
 *
 * Notes: bounds checking for pixels 2, 3 and 4 is not necessary, since pitching ensures we always have
 * multiples of at least 4.
 */
__global__ void
rgb2gray_kernel(uchar *inputImage, uchar *grayImage, const int width, const int height, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Make sure we are within bounds
	if (x * 4 >= width || y >= height) return;

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

/**
 * Function calling the rgb2gray kernel. Also initializes the device context.
 */
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

	// Allocate two device buffers (pitched)
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
	error = cudaMemcpy2D(inputImage_device, pitch, inputImage, width * sizeof(uchar), width * sizeof(uchar),
		height * 3, cudaMemcpyHostToDevice);
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
	error = cudaMemcpy2D(grayImage, width, grayImage_device, pitch, width * sizeof(uchar),
		height, cudaMemcpyDeviceToHost);
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

/**
 * Kernel which calculates a histogram of pixel values. Uses the following optimizations:
 *  - Pixels are fetched in blocks of 4 to reduce the number of memory accesses
 *  - Histogram values are first written to shared memory per block, then written atomically to global
 *    device memory atomically.
 *  - Input and output buffers are pitched to improve coalescing of the memory accesses
 */
__global__ void
histogram1D_kernel(uchar *grayImage, const int width, const int height, uint *histogram, const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fetch 4 bytes from device buffer
	const uchar4 in = ((uchar4*)grayImage)[(y * pitch / 4) + x];

	// Initialize shared histogram
	__shared__ uint histogram_shared[HISTOGRAM_SIZE];
	const int histogramIndex = blockDim.y * threadIdx.y + threadIdx.x;
	histogram_shared[histogramIndex] = 0;
	__syncthreads();

	// Make sure we are within bounds
	if (x * 4 >= width || y >= height) return;

	// Add pixel data to shared histogram
	atomicAdd(&histogram_shared[in.x], 1);
	if (x * 4 + 3 < width) {
		atomicAdd(&histogram_shared[in.y], 1);
		atomicAdd(&histogram_shared[in.z], 1);
		atomicAdd(&histogram_shared[in.w], 1);
	} else if (x * 4 + 2 < width) {
		atomicAdd(&histogram_shared[in.y], 1);
		atomicAdd(&histogram_shared[in.z], 1);
	} else if (x * 4 + 1 < width) {
		atomicAdd(&histogram_shared[in.y], 1);
	}

	// Atomically add shared histogram to global histogram
	__syncthreads();
	atomicAdd(&histogram[histogramIndex], histogram_shared[histogramIndex]);
}

/**
 * Function calling the histogram kernel.
 */
void histogram1D(uchar *grayImage, uchar *histogramImage, const int width, const int height, uint *histogram,
NSTimer &timer) {
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
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar),
		height, cudaMemcpyHostToDevice);
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
	for (uint i = 0; i < HISTOGRAM_SIZE; i++) {
		if (histogram[i] > max) {
			max = histogram[i];
		}
	}

	// Generate histogram image
	for (int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH) {
		uint value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for (uint y = 0; y < value; y++) {
			for (uint i = 0; i < BAR_WIDTH; i++) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for (uint y = value; y < HISTOGRAM_SIZE; y++) {
			for (uint i = 0; i < BAR_WIDTH; i++) {
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

/**
 * Helper function for the contrast1D kernel to determine the new value of pixel.
 */
__device__ void
contrast1D_pixelValue (uchar &pixel, const int min, const int max, const float diff) {
	float temp = 255.0f * (pixel - min) / diff;
	// Two statements below used to be branches
	temp = fminf(temp, 255.0f);
	temp = fmaxf(temp, 0.0f);
	pixel = static_cast< uchar >(temp);
}

/**
 * Kernel which improves the contrast of the image. Uses the following optimizations:
 *  - Pixels are fetched in blocks of 4 to reduce the number of memory accesses
 *  - Branches are eliminated and replaced by single-precision minimum and maximum functions
 *  - Input and output buffers are pitched to improve coalescing of the memory accesses
 *
 * Notes: bounds checking for pixels 2, 3 and 4 is not necessary, since pitching ensures we always have
 * multiples of at least 4.
 */
__global__ void
contrast1D_kernel(uchar *grayImage, const int width, const int height, const int min, const int max, const float diff,
const size_t pitch) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fetch 4 pixels from device buffer
	uchar4 pixels = ((uchar4*)grayImage)[(y * pitch / 4) + x];

	// Make sure we are within bounds
	if (x * 4 >= width || y >= height) return;

	// Calculate new pixel values for all 4 pixels
	contrast1D_pixelValue(pixels.x, min, max, diff);
	contrast1D_pixelValue(pixels.y, min, max, diff);
	contrast1D_pixelValue(pixels.z, min, max, diff);
	contrast1D_pixelValue(pixels.w, min, max, diff);

	// Store 4 pixels back to the device buffer
	((uchar4*)grayImage)[(y * pitch / 4) + x] = pixels;
}

/**
 * Function calling the constrast1D kernel.
 */
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

	while ((i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i++;
	}
	int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ((i > min) && (histogram[i] < CONTRAST_THRESHOLD)) {
		i--;
	}
	int max = i;
	float diff = max - min;

	// Copy the grayscale image from the host to the device
	copyToDeviceTime.start();
	error = cudaMemcpy2D(grayImage_device, pitch, grayImage, width * sizeof(uchar), width * sizeof(uchar),
		height, cudaMemcpyHostToDevice);
	checkError(error, "Failed to copy grayImage from host to device (error code %s)\n");
	copyToDeviceTime.stop();

	// Launch the kernel
	kernelTime.start();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(ceil((float)width / 4 / threadsPerBlock.x), ceil((float)height / threadsPerBlock.y));
	contrast1D_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, width, height, min, max, diff, pitch);
	checkError(cudaGetLastError(), "Failed to launch contrast1D_kernel (error code %s)\n");
	cudaDeviceSynchronize();
	kernelTime.stop();

	// Copy the grayscale image from the device to the host
	copyFromDeviceTime.start();
	error = cudaMemcpy2D(grayImage, width, grayImage_device, pitch, width * sizeof(uchar),
		height, cudaMemcpyDeviceToHost);
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

/**
 * Constant variable containing the filter kernel.
 */
__constant__ float filter_constant[FILTER_SIZE][FILTER_SIZE];

/**
 * Kernel which smoothes an image using a 25-point filter. This version does bounds checking, has no
 * shared memory optimizations and should be used for the borders of the image.
 */
__global__ void
triangularSmooth_kernel_borders(uchar *grayImage, uchar *smoothImage, const int width, const int height,
const int xOffset, const int yOffset) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x + xOffset;
	const int y = blockDim.y * blockIdx.y + threadIdx.y + yOffset;

	// Make sure we are within bounds
	if (x >= width || y >= height) return;

 	uint filterItem = 0;
	float filterSum = 0.0f;
	float smoothPix = 0.0f;

	for (int fy = y - 2; fy <= y + 2; fy++) {
		for (int fx = x - 2; fx <= x + 2; fx++) {
			if (((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width))) {
				filterItem++;
				continue;
			}

			smoothPix += grayImage[(fy * width) + fx] * filter_constant[0][filterItem];
			filterSum += filter_constant[0][filterItem];
			filterItem++;
		}
	}

	smoothPix /= filterSum;
	smoothImage[(y * width) + x] = static_cast< uchar >(smoothPix);
}

/**
 * Helper function for the triangularSmooth kernel that fetches 1 pixel to shared memory.
 */
__device__ void
triangularSmooth_fetchPixel(uchar *image_shared, uchar *image, const int width, const int height,
const int x, const int y, const int xOffset, const int yOffset) {
	image_shared[(SMOOTH_BLOCK_WIDTH + 4) * (threadIdx.y + 2 + yOffset) + (threadIdx.x + 2 + xOffset)] =
		image[(y + yOffset) * width + (x + xOffset)];
}

/**
 * Kernel which smoothes an image using a 25-point filter. This version does no bounds checking and expects a border
 * of at least 2 pixels around the area to process. It uses shared memory to cache a part of the image in order to
 * reduce the number of memory accesses.
 */
__global__ void
triangularSmooth_kernel(uchar *grayImage, uchar *smoothImage, const int width, const int height) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x + 2;
	const int y = blockDim.y * blockIdx.y + threadIdx.y + 2;

	// Allocate shared memory to cache part of image
	__shared__ uchar grayImage_shared[(SMOOTH_BLOCK_WIDTH + 4) * (SMOOTH_BLOCK_HEIGHT + 4)];

	// All threads fetch their corresponding pixel (no offset)
	triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 0, 0);

	// Left 2 columns fetch 2 border pixels on left
	if (threadIdx.x < 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, -2, 0);
	}
	// Top 2 rows fetch 2 border pixels on top
	if (threadIdx.y < 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 0, -2);
	}
	// Right 2 columns fetch 2 border pixels on right
	if (threadIdx.x >= SMOOTH_BLOCK_WIDTH - 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 2, 0);
	}
	// Bottom 2 rows fetch 2 border pixels on bottom
	if (threadIdx.y >= SMOOTH_BLOCK_HEIGHT - 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 0, 2);
	}
	// Topleft 4 threads fetch 4 corner pixels
	if (threadIdx.x < 2 && threadIdx.y < 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, -2, -2);
	}
	// Topright 4 threads fetch 4 corner pixels
	if (threadIdx.x >= SMOOTH_BLOCK_HEIGHT - 2 && threadIdx.y < 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 2, -2);
	}
	// Bottomleft 4 threads fetch 4 corner pixels
	if (threadIdx.x < 2 && threadIdx.y >= SMOOTH_BLOCK_HEIGHT - 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, -2, 2);
	}
	// Bottomright 4 threads fetch 4 corner pixels
	if (threadIdx.x >= SMOOTH_BLOCK_HEIGHT - 2 && threadIdx.y >= SMOOTH_BLOCK_HEIGHT - 2) {
		triangularSmooth_fetchPixel(grayImage_shared, grayImage, width, height, x, y, 2, 2);
	}

	// Wait until the cache is filled
	__syncthreads();

	// Make sure we are within bounds
	if (x >= width - 2 || y >= height - 2) return;

	float filterSum = 0.0f;
	float smoothPix = 0.0f;

	for (int dy = 0; dy <= 4; dy++) {
		for (int dx = 0; dx <= 4; dx++) {
			smoothPix += grayImage_shared[(threadIdx.y + dy) * (SMOOTH_BLOCK_WIDTH + 4) + (threadIdx.x + dx)]
				* filter_constant[dy][dx];
			filterSum += filter_constant[dy][dx];
		}
	}

	smoothPix /= filterSum;
	smoothImage[(y * width) + x] = static_cast< uchar >(smoothPix);
}

/**
 * Function calling the triangularSmooth kernels.
 */
void triangularSmooth(uchar *grayImage, uchar *smoothImage, const int width, const int height, const float *filter,
NSTimer &timer) {
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

	// Launch the different kernels
	kernelTime.start();
	int widthProcessed = 0, heightProcessed = 0;
	{
		// Main image area, fast kernel
		dim3 threadsPerBlock(SMOOTH_BLOCK_WIDTH, SMOOTH_BLOCK_HEIGHT);
		dim3 blocksPerGrid(floor((float)(width - 4) / threadsPerBlock.x),
			floor((float)(height - 4) / threadsPerBlock.y));
		triangularSmooth_kernel<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device,
			width, height);
		checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel (error code %s)\n");
		widthProcessed += blocksPerGrid.x * threadsPerBlock.x;
		heightProcessed += blocksPerGrid.y * threadsPerBlock.y;
	}
	{
		// Left border (width 2), bounds-checking kernel
		dim3 threadsPerBlock(2, 256);
		dim3 blocksPerGrid(1, ceil((float)height / threadsPerBlock.y));
		triangularSmooth_kernel_borders<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device,
			width, height, 0, 0);
		checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel_device (error code %s)\n");
		widthProcessed += blocksPerGrid.x * threadsPerBlock.x;
	}
	{
		// Top border (height 2), bounds-checking kernel
		dim3 threadsPerBlock(256, 2);
		dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), 1);
		triangularSmooth_kernel_borders<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device,
			width, height, 0, 0);
		checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel_device (error code %s)\n");
		heightProcessed += blocksPerGrid.y * threadsPerBlock.y;
	}
	{
		// Right border (width variable), bounds-checking kernel
		int blockWidth = width - widthProcessed;
		dim3 threadsPerBlock(blockWidth, 512 / blockWidth);
		dim3 blocksPerGrid(1, ceil((float)height / threadsPerBlock.y));
		triangularSmooth_kernel_borders<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device,
			width, height, widthProcessed, 0);
		checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel_device (error code %s)\n");
	}
	{
		// Bottom border (height variable), bounds-checking kernel
		int blockHeight = height - heightProcessed;
		dim3 threadsPerBlock(512 / blockHeight, blockHeight);
		dim3 blocksPerGrid(ceil((float)width / threadsPerBlock.x), 1);
		triangularSmooth_kernel_borders<<<blocksPerGrid, threadsPerBlock>>>(grayImage_device, smoothImage_device,
			width, height, 0, heightProcessed);
		checkError(cudaGetLastError(), "Failed to launch triangularSmooth_kernel_device (error code %s)\n");
	}
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
	cout << "triangularSmooth (allocation): \t\t" << allocationTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (copyToDevice): \t" << copyToDeviceTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cout << "triangularSmooth (copyFromDevice): \t" << copyFromDeviceTime.getElapsed() << " seconds." << endl;
}

