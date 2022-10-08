#ifndef IMAGE_H
#define IMAGE_H

#include "darknet.h"


image make_empty_image(int w, int h, int c);
image make_image(int w, int h, int c);
image load_image_data(int* w, int* h);

#endif