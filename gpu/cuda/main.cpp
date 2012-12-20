
#include <CImg.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "defines.h"

using cimg_library::CImg;
using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

extern void rgb2gray(uchar *inputImage, uchar *grayImage, const int width, const int height, NSTimer &timer);
extern void histogram1D(uchar *grayImage, uchar *histogramImage, const int width, const int height, uint *histogram,
	NSTimer &timer);
extern void contrast1D(uchar *grayImage, const int width, const int height, uint *histogram, NSTimer &timer);
extern void triangularSmooth(uchar *grayImage, uchar *smoothImage, const int width, const int height,
	const float *filter, NSTimer &timer);


int main(int argc, char *argv[]) {
	NSTimer total = NSTimer("total", false, false);

	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< uchar > inputImage = CImg< uchar >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	// Image is loaded, start timing
	total.start();

	// Convert the input image to grayscale 
	CImg< uchar > grayImage = CImg< uchar >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height(), total);
	total.stop();
	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./grayscale.bmp");
	}
	total.start();
	
	// Compute 1D histogram
	CImg< uchar > histogramImage = CImg< uchar >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	uint *histogram = new uint [HISTOGRAM_SIZE];

	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, total);
	total.stop();
	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./histogram.bmp");
	}
	total.start();

	// Contrast enhancement
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, total);
	total.stop();
	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./contrast.bmp");
	}
	total.start();

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< uchar > smoothImage = CImg< uchar >(grayImage.width(), grayImage.height(), 1, 1);

	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter, total);
	// Job done, stop timing
	total.stop();
	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}
	smoothImage.save("./smooth.bmp");

	// Wrap up
	cout << fixed << setprecision(6) << endl;
	cout << "Execution time: \t\t\t" << total.getElapsed() << " seconds." << endl << endl;
	
	return 0;
}

