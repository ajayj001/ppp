#ifndef  DEFINES_H
#define  DEFINES_H

// Typedefs
typedef unsigned int uint;
typedef unsigned char uchar;

// Constants
const bool displayImages = false;
const bool saveAllImages = false;
const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;
const unsigned int FILTER_SIZE = 5;
const float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
/*const float filter[][] = {	{1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
							{1.0f, 2.0f, 2.0f, 2.0f, 1.0f},
							{1.0f, 2.0f, 3.0f, 2.0f, 1.0f},
							{1.0f, 2.0f, 2.0f, 2.0f, 1.0f},
							{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}	};*/


// rgb2gray parameters
#define RED_COEFFICIENT   0.30f
#define GREEN_COEFFICIENT 0.59f
#define BLUE_COEFFICIENT  0.11f

// rectangularSmooth parameters
#define SMOOTH_BLOCK_WIDTH  16
#define SMOOTH_BLOCK_HEIGHT 16

#endif