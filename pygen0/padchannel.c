#include <stdio.h>
#include <math.h>
#include "D:/lab/ZYNQ_YOLO/runall/pygen0/input_img_sort.h"

void main()
{
    FILE* fp3;
    fp3 = fopen("D:/lab/ZYNQ_YOLO/runall/pygen0/group_0_input.h", "w");
    fprintf(fp3, "#ifndef LAYER_INPUT_H\n");
    fprintf(fp3, "#define LAYER_INPUT_H\n\n");
    fprintf(fp3, "short layer_input[]={");


    int i,j;
    for (i = 0; i < 173056; i++)
    {
        for (j = 0; j < 3; j++)
        {
            fprintf(fp3, "%hd,\n", SizedData_tranfp[i * 3 + j]);
        }
        if(i < 173055){
            fprintf(fp3, "0,\n");
        }
        else
        fprintf(fp3, "0\n");

    }

    fseek(fp3, -2, SEEK_CUR);
    fprintf(fp3, "};\n");
    fprintf(fp3, "#endif\n");
    fclose(fp3);
}