#include "image.h"
//#include "D:/lab/ZYNQ_YOLO/image_load/input_image.h"
#include "input_image_short.h"

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (fp_data_type*)calloc(h * w * c, sizeof(float));
	
	return out;
}

image load_image_data(int* w, int* h)
{
	image im = make_image(sized_w, sized_h, 3);
	*w = sized_w;
	*h = sized_h;

	//fprintf(stderr,"%d\n\r",sizeof(SizedData)/sizeof(float));

//	for (int k = 0; k < 3; ++k) {
//		for (int j = 0; j < sized_h; ++j) {
//			for (int i = 0; i < sized_w; ++i) {
//				int dst_index = i + sized_w * j + sized_w * sized_h * k;
//				//fprintf(stderr,"%d,%d,%d\n\r",j,i,dst_index);
//				im.data[dst_index] = SizedData[dst_index];
//			}
//		}
//	}
	memcpy(im.data,SizedData,sizeof(SizedData));
	return im;
}
