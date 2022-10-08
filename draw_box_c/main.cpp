#include <stdio.h>
#include "darknet.h"
#include "coco_names.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void free_image(image m)
{
    if (m.data) {
        free(m.data);
    }
}

static int entry_index0(int batch, int location, int entry)
{
    int n = location / (13 * 13);
    int loc = location % (13 * 13);
    return batch * 43095 + n * 13 * 13 * (4 + 80 + 1) + entry * 13 * 13 + loc;
}
static int entry_index(int batch, int location, int entry)
{
    int n = location / (26 * 26);
    int loc = location % (26 * 26);
    return batch * 172380 + n * 26 * 26 * (4 + 80 + 1) + entry * 26 * 26 + loc;
}

int yolo_num_detections(float thresh)
{
	float *output=(float *)calloc(1*3*26*26*85, sizeof(float));

    int i, j, n;
    int count = 0;
    FILE* fpread;
    fpread = fopen("D:/lab/ZYNQ_YOLO/draw_box/src/yolo1_cal.dat", "r");
    if (fpread == NULL)
    {
        printf("file is error.");
        return -1;
    }

    for (j = 0; j < 85 * 3 * 26 * 26; j++)
    {
    	fscanf(fpread, "%f", output + j);
    }


    fclose(fpread);
    for (i = 0; i < 26 * 26; ++i) {
        for (n = 0; n < 3; ++n) {
            int obj_index = entry_index(0, n * 26 * 26 + i, 4);
//            fprintf(stderr, "output1[%d]=%f",obj_index,output1[obj_index]);
//            std::cout<<"output["<<obj_index<<"]="<<output[obj_index]<<"\n";
            if (output[obj_index] > thresh) {
                ++count;

            }
        }
    }
    fprintf(stderr, "count=%d",count);
    return count;
}

int yolo_num_detections0(float thresh)
{
	float *output=(float *)calloc(1*3*13*13*85, sizeof(float));

    int i, j, n;
    int count = 0;
    FILE* fpread;
    fpread = fopen("D:/lab/ZYNQ_YOLO/draw_box/src/yolo0_cal.dat", "r");
    if (fpread == NULL)
    {
        printf("file is error.");
        return -1;
    }

    for (j = 0; j < 85 * 3 * 13 * 13; j++)
    {
    	fscanf(fpread, "%f", output + j);
    }


    fclose(fpread);
    for (i = 0; i < 13 * 13; ++i) {
        for (n = 0; n < 3; ++n) {
            int obj_index = entry_index0(0, n * 13 * 13 + i, 4);
//            fprintf(stderr, "output1[%d]=%f",obj_index,output1[obj_index]);
//            std::cout<<"output["<<obj_index<<"]="<<output[obj_index]<<"\n";
            if (output[obj_index] > thresh) {
                ++count;

            }
        }
    }
    fprintf(stderr, "count=%d",count);
    return count;
}

int num_detections(float thresh,int yolo_cnt)
{
    int i;
    int s = 0;
    if(yolo_cnt==0)
    	s += yolo_num_detections0(thresh);
    else
    	s += yolo_num_detections(thresh);
    return s;
}

detection* make_network_boxes(float thresh, int* num,int yolo_cnt)
{
    int i;
    int nboxes = num_detections(thresh,yolo_cnt);
    int coords = 0;
    if (num) *num = nboxes;
    detection* dets = (detection *)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float *)calloc(80, sizeof(float));
        if (coords > 4) {
            dets[i].mask = (float *)calloc(coords - 4, sizeof(float));
        }
    }
    return dets;
}

box get_yolo_box(float * x, fp_weight_type *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    short biases_true[2];
    memcpy(biases_true,&biases[2*n],2*sizeof(short));
    b.x = (i + (float)x[index + 0 * stride]) / lw;
    b.y = (j + (float)x[index + 1 * stride]) / lh;
    b.w = exp((float)x[index + 2 * stride]) * biases_true[0] / w;
    b.h = exp((float)x[index + 3 * stride]) * biases_true[1] / h;
    printf("b.x=%f,b.y=%f,b.w=%f,b.h=%f",b.x,b.y,b.w,b.h);
    return b;
}

void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int get_yolo_detections0(int w, int h, int netw, int neth, float thresh, int* map, int relative, detection* dets)
{
    int i, j, n;
    float *output=(float *)calloc(1*3*13*13*85, sizeof(float));

    short anchors[12] = {10,14,23,27,37,58,81,82,135,169,344,319};
    fp_weight_type *biases = (fp_weight_type *)calloc(6*2, sizeof(fp_weight_type));
    memcpy(biases,anchors,2*6*sizeof(short));

    int *mask = (int *)calloc(3, sizeof(int));
	mask[0] = 3;
	mask[1] = 4;
	mask[2] = 5;

    FILE* fpread;
    fpread = fopen("D:/lab/ZYNQ_YOLO/draw_box/src/yolo0_cal.dat", "r");
    if (fpread == NULL)
    {
        printf("file is error.");
        return -1;
    }

    for (j = 0; j < 85 * 3 * 13 * 13; j++)
    {
    	fscanf(fpread, "%f", output + j);
    }


    fclose(fpread);
    float* predictions = output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < 13 * 13; ++i) {
        int row = i / 13;
        int col = i % 13;


        for (n = 0; n < 3; ++n) {
            int obj_index = entry_index0(0, n * 13 * 13 + i, 4);
            float objectness = predictions[obj_index];
//            fprintf(stderr,"i=%d,n=%d,objectness=%f\n",i,n,objectness);
            if (objectness <= thresh) continue;
            int box_index = entry_index0(0, n * 13 * 13 + i, 0);
//            fprintf(stderr,"box_index=%d",box_index);
            dets[count].bbox = get_yolo_box(predictions, biases, mask[n], box_index, col, row, 13, 13, netw, neth, 13 * 13);
//            printf("yolo0 l.mask[%d]=%d\n",n,mask[n]);
            dets[count].objectness = objectness;
            dets[count].classes = 80;
            for (j = 0; j < 80; ++j) {
                int class_index = entry_index0(0, n * 13 * 13 + i, 4 + 1 + j);
//                fprintf(stderr,"i=%d,n=%d,j=%d,class_index=%d\n",i,n,j,class_index);
                float prob = objectness * (float)predictions[class_index];
                fprintf(stderr,"obj_index=%d,class_index=%d\n",obj_index,class_index);
                fprintf(stderr,"count=%d,objectness=%f,predictions=%f,prob=%f\n",count,objectness,predictions[class_index],prob);
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
                fprintf(stderr,"dets[%d].prob[%d]=%f\n",count,j,dets[count].prob[j]);
            }
            ++count;
        }
    }
//    fprintf(stderr,"count=%d\n",count);
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

int get_yolo_detections(int w, int h, int netw, int neth, float thresh, int* map, int relative, detection* dets)
{
    int i, j, n;
    float *output=(float *)calloc(1*3*26*26*85, sizeof(float));

    short anchors[12] = {10,14,23,27,37,58,81,82,135,169,344,319};
    fp_weight_type *biases = (fp_weight_type *)calloc(6*2, sizeof(fp_weight_type));
    memcpy(biases,anchors,2*6*sizeof(short));

    int *mask = (int *)calloc(3, sizeof(int));
	mask[0] = 0;
	mask[1] = 1;
	mask[2] = 2;
    FILE* fpread;
    fpread = fopen("D:/lab/ZYNQ_YOLO/draw_box/src/yolo1_cal.dat", "r");
    if (fpread == NULL)
    {
        printf("file is error.");
        return -1;
    }

    for (j = 0; j < 85 * 3 * 26 * 26; j++)
    {
    	fscanf(fpread, "%f", output + j);
    }


    fclose(fpread);
    float* predictions = output;
    //if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;

    for (i = 0; i < 26 * 26; ++i) {
        int row = i / 26;
        int col = i % 26;


        for (n = 0; n < 3; ++n) {
            int obj_index = entry_index(0, n * 26 * 26 + i, 4);
            float objectness = predictions[obj_index];
            fprintf(stderr,"i=%d,n=%d,objectness=%f\n",i,n,objectness);
            if (objectness <= thresh) continue;
            int box_index = entry_index(0, n * 26 * 26 + i, 0);
//            fprintf(stderr,"box_index=%d",box_index);
            dets[count].bbox = get_yolo_box(predictions, biases,mask[n], box_index, col, row, 26, 26, netw, neth, 26 * 26);
            printf("yolo1 l.mask[%d]=%d\n",n,mask[n]);
            dets[count].objectness = objectness;
            dets[count].classes = 80;
            for (j = 0; j < 80; ++j) {
                int class_index = entry_index(0, n * 26 * 26 + i, 4 + 1 + j);
//                fprintf(stderr,"i=%d,n=%d,j=%d,class_index=%d\n",i,n,j,class_index);
                float prob = objectness * (float)predictions[class_index];
//                fprintf(stderr,"prob=%f\n",prob);
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
//    fprintf(stderr,"count=%d\n",count);
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}



void fill_network_boxes(int w, int h, float thresh, float hier, int* map, int relative, detection* dets,int yolo_cnt)
{
	int count;
	if(yolo_cnt == 0)
		count = get_yolo_detections0(w, h, 416,416, thresh, map, relative, dets);
	else
		count = get_yolo_detections(w, h, 416,416, thresh, map, relative, dets);
    dets += count;
}

detection* get_network_boxes(int w, int h, float thresh, float hier, int* map, int relative, int* num,int yolo_cnt)
{
    detection* dets = make_network_boxes(thresh, num,yolo_cnt);
    fill_network_boxes(w, h, thresh, hier, map, relative, dets,yolo_cnt);
    return dets;
}

int nms_comparator(const void* pa, const void* pb)
{
    detection a = *(detection*)pa;
    detection b = *(detection*)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

image load_image_stb(char* filename, int channels)
{
    int w = 0;
    int h = 0;
    int c = 0;
    image im = make_image(w, h, c);
    unsigned char* data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + c * i + c * w * j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}
image load_image(char* filename, int w, int h, int c)
{
    image out = load_image_stb(filename, c);

    return out;
}

image load_image_color(char* filename, int w, int h)
{
    return load_image(filename, w, h, 3);
}

image** load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image** alphabets = (image**)calloc(nsize, sizeof(image));
    for (j = 0; j < nsize; ++j) {
        alphabets[j] = (image*)calloc(128, sizeof(image));
        for (i = 32; i < 127; ++i) {
            char buff[256];
            sprintf(buff, "D:/lab/YOLO_FPGA/Tiny_YOLO_v3_ZYNQ-master/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if (x1 < 0) x1 = 0;
    if (x1 >= a.w) x1 = a.w - 1;
    if (x2 < 0) x2 = 0;
    if (x2 >= a.w) x2 = a.w - 1;

    if (y1 < 0) y1 = 0;
    if (y1 >= a.h) y1 = a.h - 1;
    if (y2 < 0) y2 = 0;
    if (y2 >= a.h) y2 = a.h - 1;

    for (i = x1; i <= x2; ++i) {
        a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
        a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

        a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
        a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

        a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
        a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
    }
    for (i = y1; i <= y2; ++i) {
        a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
        a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

        a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
        a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

        a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
        a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int width, float r, float g, float b)
{
    int i;
    for (i = 0; i < width; ++i) {
        draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

image copy_image(image p)
{
    image copy = p;
    copy.data = (fp_data_type*)calloc(p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
    return copy;
}
void fill_cpu(int N, fp_data_type ALPHA, fp_data_type* X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i) X[i * INCX] = ALPHA;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w&& y < m.h&& c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w&& y < m.h&& c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] = val;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x, y, k;
    for (k = 0; k < source.c; ++k) {
        for (y = 0; y < source.h; ++y) {
            for (x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                set_pixel(dest, dx + x, dy + y, k, val);
            }
        }
    }
}

static float get_pixel_extend(image m, int x, int y, int c)
{
    if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
    /*
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    */
    if (c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}

void composite_image(image source, image dest, int dx, int dy)
{
    int x, y, k;
    for (k = 0; k < source.c; ++k) {
        for (y = 0; y < source.h; ++y) {
            for (x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx + x, dy + y, k);
                set_pixel(dest, dx + x, dy + y, k, val * val2);
            }
        }
    }
}

image tile_images(image a, image b, int dx)
{
    if (a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w * c.h * c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0);
    composite_image(b, c, a.w + dx, 0);
    return c;
}

image border_image(image a, int border)
{
    image b = make_image(a.w + 2 * border, a.h + 2 * border, a.c);
    int x, y, k;
    for (k = 0; k < b.c; ++k) {
        for (y = 0; y < b.h; ++y) {
            for (x = 0; x < b.w; ++x) {
                float val = get_pixel_extend(a, x - border, y - border, k);
                if (x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

image get_label(image** characters, char* string, int size)
{
    size = size / 10;
    if (size > 7) size = 7;
    image label = make_empty_image(0, 0, 0);
    while (*string) {
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h * .25);
    free_image(label);
    return b;
}

void draw_label(image a, int r, int c, image label, const float* rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for (j = 0; j < h && j + r < a.h; ++j) {
        for (i = 0; i < w && i + c < a.w; ++i) {
            for (k = 0; k < label.c; ++k) {
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i + c, j + r, k, rgb[k] * val);
            }
        }
    }
}

image float_to_image(int w, int h, int c, float* data)
{
    image out = make_empty_image(w, h, c);
    out.data = (fp_data_type*)data;
    return out;
}

static void add_pixel(image m, int x, int y, int c, fp_data_type val)
{
    assert(x < m.w&& y < m.h&& c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] += val;
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c * w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image threshold_image(image im, float thresh)
{
    int i;
    image t = make_image(im.w, im.h, im.c);
    for (i = 0; i < im.w * im.h * im.c; ++i) {
        t.data[i] = im.data[i] > thresh ? 1 : 0;
    }
    return t;
}
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    //printf("%f\n", r);
    return r;
}

void draw_detections(image im, detection* dets, int num, float thresh, char** names, int classes)
{
//    fprintf(stderr, "Begin of Detection\n\r");
	int i, j;
    image** alphabet = load_alphabet();
//    fprintf(stderr, "num = %d",num);
    for (i = 0; i < num; ++i) {
        char labelstr[4096] = { 0 };
        int class_idx = -1;
        for (j = 0; j < classes; ++j) {
//        	fprintf (stderr,"%f",dets[i].prob[j]);
            if (dets[i].prob[j] > thresh) {
                if (class_idx < 0) {
                    strcat(labelstr, names[j]);
                    class_idx = j;
                }
                else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                fprintf (stderr,"%s: %.0f%%", names[j], dets[i].prob[j] * 100);
            }
        }
        printf("class_idx=%d",class_idx);
        if (class_idx >= 0) {
            int width = im.h * .006;
            int offset = class_idx * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];
            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;

            box b = dets[i].bbox;

//            printf("b.x=%d,b.w=%d,b.y=%d,b.h=%d", b.x, b.w,b.y,b.h);
            int left = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top = (b.y - b.h / 2.) * im.h;
            int bot = (b.y + b.h / 2.) * im.h;


            if (left < 0) left = 0;
            if (right > im.w - 1) right = im.w - 1;
            if (top < 0) top = 0;
            if (bot > im.h - 1) bot = im.h - 1;
            fprintf(stderr,"  (l:%d,r:%d,t:%d,b:%d)\n", left, right, top, bot);

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, labelstr, (im.h * .03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (dets[i].mask) {
                image mask = float_to_image(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image(mask, b.w * im.w, b.h * im.h);
                image tmask = threshold_image(resized_mask, .5);
                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }

        }
    }
}

void free_detections(detection* dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

void save_image_options(image im, const char* name, IMTYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if (f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char* data = (unsigned char*)calloc(im.w * im.h * im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < im.w * im.h; ++i) {
            data[i * im.c + k] = (unsigned char)(255 * im.data[i + k * im.w * im.h]);
        }
    }
    int success = 0;
    if (f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w * im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if (!success) fprintf(stderr, "Failed to write image %s\n", buff);
}


void save_image(image im, const char* name)
{
    save_image_options(im, name, JPG, 80);
}


int main()
{
    float thresh = 0.5;
    float hier_thresh = 0.5;
    float nms = .45;
    char** names = coco_names;

    int im_w, im_h;
    image sized = load_image_data(&im_w, &im_h);
    fp_data_type* X = sized.data;
    fprintf(stderr, "Input Loaded\n\r");


    //yolo0
    int nboxes0 = 0;
    detection* dets0 = get_network_boxes(im_w, im_h, thresh, hier_thresh, 0, 1, &nboxes0,0);
    printf("nboxes0=%d",nboxes0);
    if (nms) do_nms_sort(dets0, nboxes0, 80, nms);
    draw_detections(sized, dets0, nboxes0, thresh, names, 80);
    free_detections(dets0, nboxes0);
    save_image(sized, "predictions");

    //yolo1
    int nboxes = 0;
    detection* dets = get_network_boxes(im_w, im_h, thresh, hier_thresh, 0, 1, &nboxes,1);
    printf("nboxes=%d",nboxes);
    if (nms) do_nms_sort(dets, nboxes, 80, nms);
    draw_detections(sized, dets, nboxes, thresh, names, 80);

    free_detections(dets, nboxes);

    save_image(sized, "predictions");

    free_image(sized);
    fprintf(stderr, "End of Detection\n\r");

    return 0;
}
