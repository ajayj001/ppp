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
const unsigned int FILTER_LENGTH = 25;
const float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// Tunable parameter
#define CONTRAST1D_PIXELS_PER_THREAD 1

#define HISTOGRAM_PIXELS_WIDTH 1
#define HISTOGRAM_PIXELS_HEIGHT 1

#endif