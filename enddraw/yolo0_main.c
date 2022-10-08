#include <stdio.h>
#include "yolo0.h"

int main()
{
	FILE *fp = fopen("yolo0_o.h","w");
	//FILE* fp = fopen("yolo1_o.h", "w");
	// fprintf(fp,"short layer_output_sdk[]={");

	int output_fold_factor = 16;
	int output_size0 = 13;
	int output_size = 26;

	//适用于yolo0输出
	int k,j,i;
	for (k = 0; k < 8; k++) {
		for (j = 0; j < 32; j++) {
			for (i = output_size0 * output_size0 * k; i < output_size0 * output_size0 * (k + 1); i++)
			{
				fprintf(fp, "%hd,\n", layer_output_ref[i * 32 + j]);
			}
		}
	}

	////适用于yolo1输出
	
	//for (int k = 0; k < 8; k++) {
	//	for (int j = 0; j < 32; j++) {
	//		for (int i = output_size * output_size * k; i < output_size * output_size * (k+1); i++)
	//		{
	//			fprintf(fp, "%hd,\n", layer_output_ref[i * 32 + j]);
	//		}
	//	}
	//}
	//	


	fseek(fp,-2,SEEK_CUR);
	// fprintf(fp,"};");
	fclose(fp);
	
	return 0;
}


