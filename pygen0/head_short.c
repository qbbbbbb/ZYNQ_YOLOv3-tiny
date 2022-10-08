#include <stdio.h>
#include <math.h>
#include "D:/lab/ZYNQ_YOLO/runall/pygen0/inputimg.h"

void main()
{
    FILE* fp1;
    fp1 = fopen("D:/lab/ZYNQ_YOLO/runall/enddraw/input_image_short.h", "w");

    fprintf(fp1, "#ifndef INPUT_IMAGE_SHORT_H\n");
    fprintf(fp1, "#define INPUT_IMAGE_SHORT_H\n\n");
    fprintf(fp1, "int im_w=%d;\n", im_w);
    fprintf(fp1, "int im_h=%d;\n", im_h);
    fprintf(fp1, "int sized_w=%d;\n", sized_w);
    fprintf(fp1, "int sized_h=%d;\n", sized_h);
    fprintf(fp1, "short SizedData_fp[]={");
    int i;
    for (i = 0; i < sizeof(SizedData) / sizeof(float); i++)
    {
        short a = SizedData[i] * 256;

        if (i != sizeof(SizedData) / sizeof(float) - 1)
            fprintf(fp1, "%d,\n", a);

        else
            fprintf(fp1, "%d};\n\n", a);

    }

    fprintf(fp1, "#endif\n");
    fclose(fp1);
}