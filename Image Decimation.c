#include "definitions.h"
#include "Image pyramid.h"
void image_pyramid(image *L1, image *L2)
{
	/**********************Image in BGR32*********************
	*******************parallelizable in to 3*****************/

	FILE *fp;
	fp = fopen(FILENAME, "rb");
	if (fp == NULL)
	{
		perror("Error opening file");
		exit(-1);
	}

	unsigned char *RGB_arr;
	RGB_arr = (unsigned char*)calloc(IMG_SIZE * 4, sizeof(unsigned char));

	fread(RGB_arr, sizeof(unsigned char), (IMG_SIZE * 4), fp);
	fclose(fp);

	int i, j;
	for (i = 0; i < L1_ROW; i++)
	for (j = 0; j < L1_COL; j++)
	{
		if(i COMPARE 0 || j COMPARE 0)
		{
			*(L1->blue + i * L1_COL + j) = 0;
			*(L1->green + i * L1_COL + j) = 0;
			*(L1->red + i * L1_COL + j) = 0;
		}

		if(i>0 && i<481 && j>0 && j<641)
		{
			*(L1->blue + i * L1_COL + j) = *(RGB_arr + ((i-1) * 640 * 4) + ((j-1) * 4) + 0);
			*(L1->green + i * L1_COL + j) = *(RGB_arr + ((i-1) * 640 * 4) + ((j-1) * 4) + 1);
			*(L1->red + i * L1_COL + j) = *(RGB_arr + ((i-1) * 640 * 4) + ((j-1) * 4) + 2);
		}

		else if(i>480 && j<640)
		{
			*(L1->blue + i * L1_COL + j) = 0;
			*(L1->green + i * L1_COL + j) = 0;
			*(L1->red + i * L1_COL + j) = 0;
		}
		else if(i < 480 && j>640)
		{
			*(L1->blue + i * L1_COL + j) = 0;
			*(L1->green + i * L1_COL + j) = 0;
			*(L1->red + i * L1_COL + j) = 0;
		}
		else if(i>480 && j>640)
		{
			*(L1->blue + i * L1_COL + j) = 0;
			*(L1->green + i * L1_COL + j) = 0;
			*(L1->red + i * L1_COL + j) = 0;
		}
	}
	free(RGB_arr);

	reduce(L1,L2);
}

void reduce(image* L1, image* L2)
{
    unsigned short p_rows, p_cols, c_rows, c_cols;
    unsigned int sum[3] = { 0, 0, 0 };
	unsigned short i, j , i_r, i_c, k, l;

	p_rows = L1_ROW;
	p_cols = L1_COL;

	c_rows = L2_ROW;
	c_cols = L2_COL;

	unsigned char kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	for (i = 2; i < p_rows-1; i += 2)
    {
    for (j = 2; j < p_cols-1; j += 2)
        {
            sum[0] = sum[1] = sum[2] = 0;
            for (k = 0; k < 3; k++)
            {
                for (l = 0; l < 3; l++)
                {
                    i_r = i + k - 1;
                    i_c = j + l - 1;
                    sum[0] += (unsigned int)(*(L1->red + i_r * p_cols + i_c) * kernel[k][l]);
                    sum[1] += (unsigned int)(*(L1->green + i_r * p_cols + i_c) * kernel[k][l]);
                    sum[2] += (unsigned int)(*(L1->blue + i_r * p_cols + i_c) * kernel[k][l]);
                }
            }

            k = ((i-1) / 2);
            l = ((j-1) / 2);
            if(k > 240 || l > 320)
            {
                printf("error after %u\t%u", i, j);
                exit(-1);
            }
            *(L2->red + k * c_cols + l) = (unsigned char)(sum[0] >> 4);
            *(L2->green + k * c_cols + l) = (unsigned char)(sum[1] >> 4);
            *(L2->blue + k * c_cols + l) = (unsigned char)(sum[2] >> 4);
        }
    }

    FILE *fps;
   	fps = fopen(FILENAME_L2, "wb");
   	if (fps == NULL)
    {
        perror("Error opening file");
        exit(-1);
    }

    fflush(fps);
    fwrite(L2->red, sizeof(unsigned char), (c_rows * c_cols), fps);
    fflush(fps);
    fwrite(L2->green, sizeof(unsigned char), (c_rows * c_cols), fps);
    fflush(fps);
    fwrite(L2->blue, sizeof(unsigned char), (c_rows * c_cols), fps);

    fclose(fps);
}
