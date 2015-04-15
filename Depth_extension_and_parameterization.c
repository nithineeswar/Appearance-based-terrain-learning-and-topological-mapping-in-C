#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define FRAME_SIZE 307200
#define HIST_SIZE 60
#define COMPARE ==
#define ROW 480
#define COL 640

#define FILENAME "C:\\Users\\nithine\\Documents\\MATLAB\\fp\\log.txt"
#define FILENAME1 "C:\\Users\\nithine\\Documents\\MATLAB\\FP\\Kmeans\\Training pics\\set 2\\N2\\N2_1rgb_frame.bin"

void histogram(unsigned char*, unsigned int*);
unsigned char getmax_f(float*, unsigned char);
void otsu_segmentation(unsigned short*);
void depth_extension(unsigned short*, unsigned short*);

typedef struct
{
	float x;
	float y;
	float z;
}coordinate_3D;

FILE *fpn;

int main()
{
	FILE *fp;
	fp = NULL;
	unsigned short center_depth_column[ROW], depth_lookup_table[ROW];
	int i, j, k, l;
	unsigned char navigable[ROW][COL], navigable_flipped[ROW][COL];
	coordinate_3D pixel_point, focal_point, physical_world_point;
	float alpha, numerator, denominator;                    //(scaling)
	unsigned int m_x, m_y;
	unsigned int metric_grid[50][100];

	fpn = fopen(FILENAME, "w");
	if (fpn == NULL)
	{
		perror("Error opening file");
		return(-1);
	}

	otsu_segmentation(center_depth_column);
	depth_extension(center_depth_column, depth_lookup_table);

	fp = fopen(FILENAME1, "rb");
	if (fp == NULL)
	{
		perror("Error opening file");
		return(-1);
	}

	fread(navigable, sizeof(unsigned char), FRAME_SIZE, fp);
	fclose(fp);

	/***********    Flip image   ***********/
	for (i = 0; i < 480; i++)
	for (j = 0; j < 640; j++)
		navigable_flipped[479 - i][639 - j] = navigable[i][j];


	focal_point.x = 0;
	focal_point.y = 0.0029;
	focal_point.z = 0.5;

	memset(metric_grid, (unsigned int)0, sizeof(unsigned int)* 50 * 100);

	for (i = 1; i < 480; i++)
	if (depth_lookup_table[479 - i] < 2000)
	{
		for (j = 1; j < 640; j++)
		if (navigable_flipped[i * 640 + j])
		{
			for (k = 0; k < 2; k++)
			for (l = 0; l < 2; l++)
			{
				pixel_point.x = ((float)(j + l - 319)) * 0.0000056;
				pixel_point.y = 0;
				pixel_point.z = focal_point.z + (float)(239 - (i + k)) * 0.0000056;

				numerator = powf((float)depth_lookup_table[479 - (i + k)] * 0.001, 2);
				denominator = powf(focal_point.x - pixel_point.x, 2) + powf(focal_point.y - pixel_point.y, 2) + powf(focal_point.z - pixel_point.z, 2);

				alpha = sqrtf(numerator / denominator);

				physical_world_point.x = pixel_point.x + alpha * (focal_point.x - pixel_point.x);
				physical_world_point.y = pixel_point.y + alpha * (focal_point.y - pixel_point.y);
				physical_world_point.z = pixel_point.z + alpha * (focal_point.z - pixel_point.z);

				m_y = ceilf(abs((float)50 + physical_world_point.x / 0.02));
				m_x = ceilf(abs(physical_world_point.y / 0.02 - (float)40));

				if (m_y < 101 && m_x < 51)
					metric_grid[m_x-1][m_y-1] = 1;
			}
		}
	}

	fclose(fpn);
	return 0;
}


void otsu_segmentation(unsigned short* center_depth_column)
{
	FILE *fp;
	unsigned short depth_frame[ROW][COL];
	unsigned char threshold, depth_frame_reduced[ROW][COL];
	unsigned int i, j, im_hist[HIST_SIZE];// max intensity of 10000 observed, 15000 >> is 58, hence the size of the histogram
	float p_of_intensity[HIST_SIZE], omega_k[HIST_SIZE], mean_k[HIST_SIZE], mean_t, sigma_b_sqd[HIST_SIZE];

	fp = fopen("C:\\Users\\nithine\\Documents\\MATLAB\\FP\\depth_frame.bin", "rb");
	if (fp == NULL)
	{
		perror("Error opening file");
		exit(-1);
	}

	fread(depth_frame, sizeof(unsigned short), FRAME_SIZE, fp);
	fclose(fp);

	for (i = 0; i < ROW; i++)
	for (j = 0; j < COL; j++)
		depth_frame_reduced[i][j] = depth_frame[i][j] >> 8;

	memset(im_hist, (unsigned int)0, sizeof(unsigned int)* HIST_SIZE);
	histogram(&depth_frame_reduced[0][0], im_hist);

	fprintf(fpn, "Histogram\n");
	for (i = 0; i < HIST_SIZE; i++)
		fprintf(fpn, "%d\t%d\n", i, im_hist[i]);

	/****** 32 bits*********/

	memset(p_of_intensity, (float)0, sizeof(float)* HIST_SIZE);

	for (i = 0; i < HIST_SIZE; i++)
	{
		if (im_hist[i])
			p_of_intensity[i] = (float)im_hist[i] / (float)FRAME_SIZE;
	}

	memset(omega_k, (float)0, sizeof(float)* HIST_SIZE);
	memset(mean_k, (float)0, sizeof(float)* HIST_SIZE);

	for (i = 0; i < HIST_SIZE; i++)
	{
		if (i COMPARE 0)
		{
			omega_k[i] = omega_k[i] + p_of_intensity[i]; // Cummulative sum
			mean_k[i] = mean_k[i] + (float)i * p_of_intensity[i];
		}
		else
		{
			omega_k[i] = omega_k[i - 1] + p_of_intensity[i];
			mean_k[i] = mean_k[i - 1] + (float)i * p_of_intensity[i]; // Cummulative sum
		}
	}

	mean_t = mean_k[i - 1];
	memset(sigma_b_sqd, (float)0, sizeof(float)* HIST_SIZE);

	for (i = 0; i < HIST_SIZE; i++)
		sigma_b_sqd[i] = powf((mean_t * omega_k[i] - mean_k[i]), (float)2) / (omega_k[i] * (1 - omega_k[i]));

	threshold = (unsigned char)getmax_f(sigma_b_sqd, (unsigned char)HIST_SIZE);
	fprintf(fpn, "Threshold %d\n", threshold);

	/**************Depth parameterization **************/

	if (threshold < mean_t)
	{
		for (i = 0; i < 480; i++)
		{
			if (depth_frame_reduced[i][320] < threshold)
				center_depth_column[i] = 0;
			else
				center_depth_column[i] = depth_frame[i][320];
		}
	}
	else
	{
		for (i = 0; i < 480; i++)
		{
			if (depth_frame_reduced[i][320] > threshold)
				center_depth_column[i] = 0;
			else
				center_depth_column[i] = depth_frame[i][320];
		}
	}

	fprintf(fpn, "Parameterized depth(center_depth_column)\n");
	for (i = 0; i < 480; i++)
		fprintf(fpn, "%d\n", center_depth_column[i]);

	return;
}

void histogram(unsigned char *arr, unsigned int *depth_frame_hist)
{
	unsigned int index;
	for (index = 0; index < FRAME_SIZE; index++)
		depth_frame_hist[arr[index]]++;
}

unsigned char getmax_f(float* vector, unsigned char size)
{
	unsigned char index = 0;
	float max_val = vector[index];
	unsigned char max_val_index = 0;

	for (index = 1; index < size; index++)
	{
		if (vector[index] >= max_val)
		{
			max_val = vector[index];
			max_val_index = index;
		}
	}
	return max_val_index;
}

void depth_extension(unsigned short *center_depth_column, unsigned short *depth_lookup_table)
{
	/**************Depth extension**************/

	unsigned short x[3], y[3];
	unsigned int d[3];
	short ls[3][3], ls_transpose[3][3];
	int ls_t_mul_ls[3][3], i, j, k;
	double determinant = 0;
	double ls_t_mul_ls_inverse[3][3];
	double A_ra, B_ra, C_ra;
	unsigned int sum_of_x_sqd = 0, sum_x = 0;
	double a_po, b_po, A_po, B_po;
	unsigned short y_ra[480], y_ex[480], y_po[480];
	unsigned short error_sum[3] = { 0, 0, 0 };
	unsigned char best_model_index = 0;					// (x, y) - training samples
	double denominator_ex, denominator_po, a_ex, b_ex, A_ex, B_ex;
	double   sum_log_y = 0, sum_log_x = 0, sum_log_x_sqd = 0, sum_x_mul_log_y = 0, sum_log_x_mul_log_y = 0;
	double f_rational[3] = { 0, 0, 0 };
	double ls_t_mul_d[3] = { 0, 0, 0 };

	/******************Rational Function********************/
	for (x[0] = 0; x[0] < 480; x[0]++)
	if (center_depth_column[x[0]])
		break;

	for (x[2] = 479; x[2] > 0; x[2]--)
	if (center_depth_column[x[2]])
		break;

	x[1] = (unsigned short)floorl((double)(x[0] + x[2]) / (double)2 + 0.5);

	i = 0;
	if (center_depth_column[x[1]]);
	else while (!center_depth_column[x[1]])
	{
		if (center_depth_column[x[1] + i])
		{
			x[1] += i;
			break;
		}
		else if (center_depth_column[x[1] - i])
		{
			x[1] -= i;
			break;
		}
		++i;
	}

	for (i = 0; i < 3; i++)
	{
		y[i] = center_depth_column[x[i]];
		d[i] = y[i] + (x[i] + 1) * y[i];
	}

	for (i = 0; i < 3; i++)
	{
		ls[i][0] = ls_transpose[0][i] = (x[i] + 1);
		ls[i][1] = ls_transpose[1][i] = 1;
		ls[i][2] = ls_transpose[2][i] = -1 * (int)y[i];

	}

	memset(ls_t_mul_ls, (int)0, sizeof(int)* 9);
	memset(ls_t_mul_ls_inverse, (double)0, sizeof(double)* 9);

	for (i = 0; i < 3; i++)
	for (j = 0; j < 3; j++)
	for (k = 0; k < 3; k++)
		ls_t_mul_ls[i][j] += ls_transpose[i][k] * ls[k][j]; // symmetric matrix

	determinant += (double)ls_t_mul_ls[0][0] * (((double)ls_t_mul_ls[1][1] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[1][2] * (double)ls_t_mul_ls[2][1]));
	determinant += (double)ls_t_mul_ls[0][1] * (double)-1 * (((double)ls_t_mul_ls[1][0] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[1][2] * (double)ls_t_mul_ls[2][0]));
	determinant += (double)ls_t_mul_ls[0][2] * (((double)ls_t_mul_ls[1][0] * (double)ls_t_mul_ls[2][1]) - ((double)ls_t_mul_ls[1][1] * (double)ls_t_mul_ls[2][0]));

	if (determinant)
	{
		ls_t_mul_ls_inverse[0][0] = (((double)ls_t_mul_ls[1][1] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[1][2] * (double)ls_t_mul_ls[2][1])) / (double)determinant;
		ls_t_mul_ls_inverse[0][1] = (double)-1 * (((double)ls_t_mul_ls[1][0] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[1][2] * (double)ls_t_mul_ls[2][0])) / (double)determinant;
		ls_t_mul_ls_inverse[0][2] = (((double)ls_t_mul_ls[1][0] * (double)ls_t_mul_ls[2][1]) - ((double)ls_t_mul_ls[1][1] * (double)ls_t_mul_ls[2][0])) / (double)determinant;
		ls_t_mul_ls_inverse[1][0] = (double)-1 * (((double)ls_t_mul_ls[0][1] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[0][2] * (double)ls_t_mul_ls[2][1])) / (double)determinant;
		ls_t_mul_ls_inverse[1][1] = (((double)ls_t_mul_ls[0][0] * (double)ls_t_mul_ls[2][2]) - ((double)ls_t_mul_ls[0][2] * (double)ls_t_mul_ls[2][0])) / (double)determinant;
		ls_t_mul_ls_inverse[1][2] = (double)-1 * (((double)ls_t_mul_ls[0][0] * (double)ls_t_mul_ls[2][1]) - (double)ls_t_mul_ls[0][1] * (double)(ls_t_mul_ls[2][0])) / (double)determinant;
		ls_t_mul_ls_inverse[2][0] = (((double)ls_t_mul_ls[0][1] * (double)ls_t_mul_ls[1][2]) - ((double)ls_t_mul_ls[0][2] * (double)ls_t_mul_ls[1][1])) / (double)determinant;
		ls_t_mul_ls_inverse[2][1] = (double)-1 * (((double)ls_t_mul_ls[0][0] * (double)ls_t_mul_ls[1][2]) - ((double)ls_t_mul_ls[0][2] * (double)ls_t_mul_ls[1][0])) / (double)determinant;
		ls_t_mul_ls_inverse[2][2] = (((double)ls_t_mul_ls[0][0] * (double)ls_t_mul_ls[1][1]) - ((double)ls_t_mul_ls[0][1] * (double)ls_t_mul_ls[1][0])) / (double)determinant;
	}

	for (i = 0; i < 3; i++)
	for (j = 0; j < 3; j++)
		ls_t_mul_d[i] += (double)ls_transpose[i][j] * (double)d[j];

	for (i = 0; i < 3; i++)
	for (j = 0; j < 3; j++)
		f_rational[i] += ls_t_mul_ls_inverse[i][j] * (double)ls_t_mul_d[j];

	A_ra = f_rational[0];
	B_ra = f_rational[1];
	C_ra = f_rational[2];

	fprintf(fpn, "\nRataional function paraneters\n");
	fprintf(fpn, "A->%f\tB->%f\tC->%f\n", f_rational[0], f_rational[1], f_rational[2]);

	/******************Exponential Function*******************/
	for (i = x[0]; (int)i < (int)(x[2] + 1); i++)
	{
		sum_of_x_sqd += (i + 1) * (i + 1);
		sum_x += (i + 1);

		sum_log_x += log((double)(i + 1));
		sum_log_x_sqd += log((double)(i + 1)) * log((double)(i + 1));
		if (center_depth_column[i])// as log 0 = -inf
		{
			sum_log_y += log((double)center_depth_column[i]);
			sum_x_mul_log_y += (double)(i + 1) * log((double)center_depth_column[i]);
			sum_log_x_mul_log_y += log((double)(i + 1)) * log((double)center_depth_column[i]);
		}
	}

	denominator_ex = ((double)(x[2] - x[0] + 1) * (double)sum_of_x_sqd) - ((double)sum_x * (double)sum_x);
	a_ex = (((double)sum_log_y * (double)sum_of_x_sqd) - ((double)sum_x * (double)sum_x_mul_log_y)) / denominator_ex;
	b_ex = (((double)(x[2] - x[0] + 1) * (double)sum_x_mul_log_y) - ((double)sum_x* (double)sum_log_y)) / denominator_ex;
	A_ex = exp(a_ex);
	B_ex = b_ex;

	fprintf(fpn, "\nExponential function paraneters\n");
	fprintf(fpn, "A->%f\tB->%f\n", A_ex, B_ex);

	/******************Polynomial Function*******************/

	denominator_po = ((double)(x[2] - x[0] + 1) * (double)sum_log_x_sqd) - ((double)sum_log_x * (double)sum_log_x);
	b_po = (((double)(x[2] - x[0] + 1) * sum_log_x_mul_log_y) - (sum_log_x * sum_log_y)) / denominator_po;
	a_po = (sum_log_y - (b_po * sum_log_x)) / (double)(x[2] - x[0] + 1);

	A_po = exp(a_po);
	B_po = b_po;

	fprintf(fpn, "\nPolynomial function paraneters\n");
	fprintf(fpn, "A->%f\tB->%f\n", A_po, B_po);

	memset(y_ra, (unsigned short)0, sizeof(unsigned short)& 480);
	memset(y_ex, (unsigned short)0, sizeof(unsigned short)& 480);
	memset(y_po, (unsigned short)0, sizeof(unsigned short)& 480);

	for (i = x[0]; (int)i < (int)(x[2] + 1); i++)
	{
		y_ra[i] = (unsigned short)floorl((A_ra * (double)(i + 1) + B_ra) / ((double)(i + 1) + C_ra) + 0.5);
		y_ex[i] = (unsigned short)floorl(A_ex * exp(B_ex * (double)(i + 1)) + 0.5);
		y_po[i] = (unsigned short)floorl(A_po * powl((double)(i + 1), B_po));
	}

	// Best fit, index(0)-> rational, index(1)-> exponential, index(2)-> polynomial
	for (i = x[0]; (int)i < (int)(x[2] + 1); i++)
	{
		error_sum[0] += abs(center_depth_column[i] - y_ra[i]);
		error_sum[1] += abs(center_depth_column[i] - y_ex[i]);
		error_sum[2] += abs(center_depth_column[i] - y_po[i]);
	}

	if (error_sum[1] < error_sum[0] && error_sum[1] < error_sum[2])
		best_model_index = 1;
	if (error_sum[2] < error_sum[0] && error_sum[2] < error_sum[1])
		best_model_index = 2;

	for (i = 0; i < x[0]; i++)
	{
		y_ra[i] = (unsigned short)floorl((A_ra * (double)(i + 1) + B_ra) / ((double)(i + 1) + C_ra) + 0.5);
		y_ex[i] = (unsigned short)floorl(A_ex * exp(B_ex * (double)(i + 1)) + 0.5);
		y_po[i] = (unsigned short)floorl(A_po * powl((double)(i + 1), B_po));
	}

	switch (best_model_index)
	{
	case 0:
		for (i = 0; i < 480; i++)
			depth_lookup_table[i] = y_ra[i];
		break;
	case 1:
		for (i = 0; i < 480; i++)
			depth_lookup_table[i] = y_ex[i];
		break;
	case 2:
		for (i = 0; i < 480; i++)
			depth_lookup_table[i] = y_po[i];
		break;
	default:
		break;
	}

	fprintf(fpn, "\nEtended depth\n");
	for (i = 0; i < 480; i++)
		fprintf(fpn, "%d\n", depth_lookup_table[i]);

	return;
}