#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

//#define DIRNAME "C:\\Users\\nithine\\Documents\\MATLAB\\fp\\Kmeans\\Training pics\\set 2\\L2_DS_1\\"

//#define DIRNAME "E:\\Datasets\\Malaga_Parking_6L\\malaga_terrain_learning_240_320\\bin\\"
#define DIRNAME "E:\\Datasets\\Malaga_Parking_6L\\malaga_terrain_learning_480_640\\bin\\"
//#define DIRNAME "E:\\Datasets\\Malaga_Parking_6L\\malaga_terrain_learning_768_1024\\bin\\"

#define IMG_SIZE 76800
#define ROWS 240
#define COLS 320
#define ROI_ROWS 64
#define ROI_COLS 128
#define ROI_shift 13

#define REDUCE
#define REDUCE_SCORING
#ifdef REDUCE
    typedef struct
    {
        unsigned char red[480][640];
        unsigned char green[480][640];
        unsigned char blue[480][640];
    } L1;
#endif


#define HIST_SIZE 2500
#define BIT_SHFIT 3
#define COMPARE ==
#define K_MEANS 3
#define K_MEANS_ITERATION 3
#define MAX_LUT_SIZE 8

#define COLOR_TYPE float
#define COLOR_TYPE_2 double
//#define COLOR_TYPE unsigned char
//#define COLOR_TYPE_2 unsigned int

///****** Frame throughput for 240x320, 480x640, 768x1024********///
///(55/04.163) 13.2116 f/s for 240x320
///(55/12.643) 04.3502 f/s for (480x640) reduced to (240x320)
///(55/14.096) 03.9018 f/s for 480x640
///(55/56.595) 00.9718 f/s for 768x1024

typedef struct
{
	unsigned char red[ROWS][COLS];
	unsigned char green[ROWS][COLS];
	unsigned char blue[ROWS][COLS];
} image;

typedef struct
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
} color;

typedef struct
{
	unsigned short red;
	unsigned short green;
	unsigned short blue;
} color_16;

typedef struct
{
	unsigned int red;
	unsigned int green;
	unsigned int blue;
} color_32;

typedef struct
{
	double red;
	double green;
	double blue;
} color_s_d_64;

typedef struct
{
	COLOR_TYPE red;
	COLOR_TYPE green;
	COLOR_TYPE blue;
	double weight;
	double covariance_matrix_sum[3][3];
	double covariance_matrix_adj_mul_weight[3][3];
	double determinant;
} color_model;

static unsigned char COLOR_MODELS = 0;
unsigned int equals(unsigned char*, unsigned char*);
unsigned short* get_maximum_index_weight(unsigned short*);
unsigned short* get_minimum_index_weight(color_model*);
void adjoint(double*, double*, double);
void determinant(double*, double*);


int main()
{
	color_model LUT[MAX_LUT_SIZE];
	FILE *fp;
	unsigned int  i, j, k, l;
	static image img;

	unsigned char histogram_rgb[HIST_SIZE][3], pixel_cluster_index[HIST_SIZE];
	unsigned char abs_sum[3], recal_flag;
	unsigned int histogram_weight[HIST_SIZE], iterations;
	unsigned short current_roi_length, kmeans_cluster_weight[3], kmeans_cluster_weight_copy[3];
	color mean_ROI, kmeans[3], kmeans_new[3];
	color_16 variance_ROI;
	color_32 mean_sum, variance_sum, recal_mean_sum[3];
	double criterion;

	unsigned short *index_weight, *LUT_min_index_weight;
	char str[100];
    double covariance_matrix_sum[3][3][3], covariance_matrix_adj_mul_weight[3][3][3], cova_diff_adj_mul_w1w2[3][3], cova_diff[3][3];
    double cova_matrix_sum_determinant[3], cova_diff_determinant, THRESHOLD, fabsl_distance;

    unsigned char count_h;
    unsigned short horizon_sum_vector[ROWS], HORIZON, max_horizon;
    color_16 smooth_sum;
    COLOR_TYPE diff_r, diff_g, diff_b;
    COLOR_TYPE_2 weight_sum;

    unsigned int ROI_SIZE  = ROI_ROWS * ROI_COLS;

    struct dirent *dptr;
    DIR *dir;

    dir = opendir(DIRNAME);
    if(dir == NULL)
        perror(DIRNAME), exit(-1);
    dptr=readdir(dir), dptr=readdir(dir);

    while((dptr = readdir(dir)))
    {
        strcpy(str, DIRNAME);
        strcat(str, dptr->d_name);

        fp = fopen(str, "rb");
        if (fp == NULL)
            perror("File opening error"), exit(-1);

        #ifdef REDUCE
            static L1 img_l1;
            fread(img_l1.red, sizeof(unsigned char), IMG_SIZE*4, fp);
            fread(img_l1.green, sizeof(unsigned char), IMG_SIZE*4, fp);
            fread(img_l1.blue, sizeof(unsigned char), IMG_SIZE*4, fp);
            fclose(fp);

            memset(img.red, (unsigned char)0, sizeof(unsigned char) * IMG_SIZE);
            memset(img.green, (unsigned char)0, sizeof(unsigned char) * IMG_SIZE);
            memset(img.blue, (unsigned char)0, sizeof(unsigned char) * IMG_SIZE);

            unsigned char kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
            unsigned int sum[3] = { 0, 0, 0 };
            unsigned short i_r, i_c;

            for (i = 1; i < (480-1); i += 2)
            for (j = 1; j < (640-1); j += 2)
            {
                sum[0] = sum[1] = sum[2] = 0;
                for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                {
                    i_r = i + k - 1;
                    i_c = j + l - 1;
                    sum[0] += (unsigned int)(img_l1.red[i_r][i_c]* kernel[k][l]);
                    sum[1] += (unsigned int)(img_l1.green[i_r][i_c] * kernel[k][l]);
                    sum[2] += (unsigned int)(img_l1.blue[i_r][i_c] * kernel[k][l]);
                }

                k = i/2;
                l = j/2;

                img.red[k][l] = (unsigned char)(sum[0] >> 4);
                img.green[k][l] = (unsigned char)(sum[1] >> 4);
                img.blue[k][l] = (unsigned char)(sum[2] >> 4);
            }
        #else
            fread(img.red, sizeof(unsigned char), IMG_SIZE, fp);
            fread(img.green, sizeof(unsigned char), IMG_SIZE, fp);
            fread(img.blue, sizeof(unsigned char), IMG_SIZE, fp);
            fclose(fp);
        #endif // REDUCE

        memset(histogram_rgb, (unsigned char)0, sizeof(unsigned char) * HIST_SIZE * 3);
        memset(histogram_weight, (unsigned int)0, sizeof(unsigned int) * HIST_SIZE);

        //Building Histogram and calculating mean
        mean_sum.red = mean_sum.green = mean_sum.blue = 0;

        for (i = (ROWS - ROI_ROWS); i < ROWS; i++)
        for (j = ((COLS/2) - ROI_COLS/2); j < ((COLS/2) + (ROI_COLS/2)); j++)
        {
            mean_sum.red += (unsigned int)img.red[i][j];
            mean_sum.green += (unsigned int)img.green[i][j];
            mean_sum.blue += (unsigned int)img.blue[i][j];

            if (((i - (ROWS - ROI_ROWS)) + (j - ((COLS/2) - ROI_COLS/2))) COMPARE 0)
            {
                histogram_rgb[0][0] = img.red[i][j] >> BIT_SHFIT;
                histogram_rgb[0][1] = img.green[i][j] >> BIT_SHFIT;
                histogram_rgb[0][2] = img.blue[i][j] >> BIT_SHFIT;
                histogram_weight[0] += 1;
            }
            else
            for (k = 0; k < ROI_SIZE; k++)
            {
                if (((histogram_rgb[k][0]) COMPARE(img.red[i][j] >> BIT_SHFIT)) && ((histogram_rgb[k][1]) COMPARE(img.green[i][j] >> BIT_SHFIT)) && ((histogram_rgb[k][2]) COMPARE(img.blue[i][j] >> BIT_SHFIT)))
                {
                    histogram_weight[k] += 1;
                    break;
                }
                if ((histogram_rgb[k][0] + histogram_rgb[k][1] + histogram_rgb[k][2] + histogram_weight[k]) COMPARE 0)
                {
                    histogram_rgb[k][0] = img.red[i][j] >> BIT_SHFIT;
                    histogram_rgb[k][1] = img.green[i][j] >> BIT_SHFIT;
                    histogram_rgb[k][2] = img.blue[i][j] >> BIT_SHFIT;
                    histogram_weight[k] += 1;
                    break;
                }
            }
        }
        mean_ROI.red = mean_sum.red >> (unsigned int)ROI_shift;
        mean_ROI.green = mean_sum.green >> (unsigned int)ROI_shift;
        mean_ROI.blue = mean_sum.blue >> (unsigned int)ROI_shift;

        //Calcualte variance
        variance_sum.red = variance_sum.green = variance_sum.blue = 0;
        for (i = (ROWS - ROI_ROWS); i < ROWS; i++)
        for (j = ((COLS/2) - ROI_ROWS); j < ((COLS/2) + (ROI_COLS/2)); j++)
        {
            variance_sum.red += ((int)img.red[i][j] - (int)mean_ROI.red) * ((int)img.red[i][j] - (int)mean_ROI.red);
            variance_sum.green += ((int)img.green[i][j] - (int)mean_ROI.green) * ((int)img.green[i][j] - (int)mean_ROI.green);
            variance_sum.blue += ((int)img.blue[i][j] - (int)mean_ROI.blue) * ((int)img.blue[i][j] - (int)mean_ROI.blue);
        }
        mean_ROI.red = mean_ROI.red >> (unsigned int)BIT_SHFIT;
        mean_ROI.green = mean_ROI.green >> (unsigned int)BIT_SHFIT;
        mean_ROI.blue = mean_ROI.blue >> (unsigned int)BIT_SHFIT;

        variance_ROI.red = variance_sum.red >> (unsigned int)(ROI_shift);
        variance_ROI.green = variance_sum.green >> (unsigned int)(ROI_shift);
        variance_ROI.blue = variance_sum.blue >> (unsigned int)(ROI_shift);

        variance_ROI.red = variance_ROI.red >> (unsigned int)(BIT_SHFIT);
        variance_ROI.green = variance_ROI.green >> (unsigned int)(BIT_SHFIT);
        variance_ROI.blue = variance_ROI.blue >> (unsigned int)(BIT_SHFIT);

        // Uniformly Distributed initialization
        kmeans[0].red = kmeans[0].green = kmeans[0].blue = (unsigned char)0;
        kmeans[1].red = kmeans[1].green = kmeans[1].blue = (unsigned char)16;
        kmeans[2].red = kmeans[2].green = kmeans[2].blue = (unsigned char)31;

        //  Initializing one of the color models to mean of ROI
        abs_sum[0] = abs_sum[1] = abs_sum[2] = 0;
        j = 0; //Index initialization
        for (i = 0; i < 3; i++)
        {
            abs_sum[i] += abs(kmeans[i].red - mean_ROI.red);
            abs_sum[i] += abs(kmeans[i].green - mean_ROI.green);
            abs_sum[i] += abs(kmeans[i].blue - mean_ROI.blue);
            if (i>0)
            if (abs_sum[i] < abs_sum[i - 1])
                j = i;// j index of minimum difference
        }

        kmeans[j].red = mean_ROI.red;
        kmeans[j].green = mean_ROI.green;
        kmeans[j].blue = mean_ROI.blue;

        for (i = 0; i < 3; i++)
            kmeans_new[i].red = kmeans_new[i].green = kmeans_new[i].blue = 1;
//        first_flag = 1;

        /*************************Kmeans clustering**********************
        ****************************************************************/
        for (iterations = 0; iterations < K_MEANS_ITERATION; iterations++)//while (equals(&kmeans_new[0].red, &kmeans[0].red))//
        {
            kmeans_cluster_weight[0] = kmeans_cluster_weight[1] = kmeans_cluster_weight[2] = 0;

            /***Cluster Assignment**/
            for (i = 0; i < ROI_SIZE; i++)
            {
                if (histogram_weight[i])
                {
                    abs_sum[0] = abs_sum[1] = abs_sum[2] = 0;
                    pixel_cluster_index[i] = 0;
                    for (j = 0; j < 3; j++)
                    {
                        abs_sum[j] += abs(histogram_rgb[i][0] - kmeans[j].red);
                        abs_sum[j] += abs(histogram_rgb[i][1] - kmeans[j].green);
                        abs_sum[j] += abs(histogram_rgb[i][2] - kmeans[j].blue);
                        if (j>0)
                        if (abs_sum[j] < abs_sum[j - 1])
                            pixel_cluster_index[i] = j;
                    }
                    kmeans_cluster_weight[pixel_cluster_index[i]] += histogram_weight[i];
                }
                else
                {
                    current_roi_length = i;
                    break;
                }
            }

            recal_flag = 0;
            for (i = 0; i < 3; i++)
            if (kmeans_cluster_weight[i] > 0)
                recal_flag++;

            memset(recal_mean_sum, (unsigned int)0, sizeof(unsigned int)* 9);

            for (i = 0; i < current_roi_length; i++)
            {
                recal_mean_sum[pixel_cluster_index[i]].red += (unsigned int)histogram_rgb[i][0] * histogram_weight[i];
                recal_mean_sum[pixel_cluster_index[i]].green += (unsigned int)histogram_rgb[i][1] * histogram_weight[i];
                recal_mean_sum[pixel_cluster_index[i]].blue += (unsigned int)histogram_rgb[i][2] * histogram_weight[i];
            }

            for (i = 0; i < 3; i++)
            {
                if (kmeans_cluster_weight[i])
                {
                    kmeans[i].red = kmeans_new[i].red = (unsigned char)floorf((float)recal_mean_sum[i].red / (float)kmeans_cluster_weight[i] + 0.5);
                    kmeans[i].green = kmeans_new[i].green = (unsigned char)floorf((float)recal_mean_sum[i].green / (float)kmeans_cluster_weight[i] + 0.5);
                    kmeans[i].blue = kmeans_new[i].blue = (unsigned char)floorf((float)recal_mean_sum[i].blue / (float)kmeans_cluster_weight[i] + 0.5);
                }
            }
            if (recal_flag == 1)
                break; // if all the pixels are in the same cluster, skip the iteration
        }

        for (i = 0; i < 3; i++)
            kmeans_cluster_weight_copy[i] = kmeans_cluster_weight[i];

/*******************************************Covariance Matrix**************************
**************************************************************************************/
        memset(covariance_matrix_sum, (double)0, sizeof(double) * 27);

        for (i = 0; i < current_roi_length; i++)
        {
            j = pixel_cluster_index[i];
            covariance_matrix_sum[j][0][0] += ((double)(histogram_rgb[i][0] - kmeans[j].red) * (double)(histogram_rgb[i][0] - kmeans[j].red) * (double)(histogram_weight[i]));
            covariance_matrix_sum[j][1][0] += ((double)(histogram_rgb[i][0] - kmeans[j].red) * (double)(histogram_rgb[i][1] - kmeans[j].green) * (double)(histogram_weight[i]));
            covariance_matrix_sum[j][2][0] += ((double)(histogram_rgb[i][0] - kmeans[j].red) * (double)(histogram_rgb[i][2] - kmeans[j].blue) * (double)(histogram_weight[i]));

            covariance_matrix_sum[j][1][1] += ((double)(histogram_rgb[i][1] - kmeans[j].green) * (double)(histogram_rgb[i][1] - kmeans[j].green) * (double)(histogram_weight[i]));
            covariance_matrix_sum[j][1][2] += ((double)(histogram_rgb[i][1] - kmeans[j].green) * (double)(histogram_rgb[i][2] - kmeans[j].blue) * (double)(histogram_weight[i]));
            covariance_matrix_sum[j][2][2] += ((double)(histogram_rgb[i][2] - kmeans[j].blue) * (double)(histogram_rgb[i][2] - kmeans[j].blue) * (double)(histogram_weight[i]));

            covariance_matrix_sum[j][0][1] = covariance_matrix_sum[j][1][0];
            covariance_matrix_sum[j][0][2] = covariance_matrix_sum[j][2][0];
            covariance_matrix_sum[j][2][1] = covariance_matrix_sum[j][1][2];
        }


        cova_matrix_sum_determinant[0] = cova_matrix_sum_determinant[1] = cova_matrix_sum_determinant[2] = (double)0;

        for (k = 0; k < 3; k++)
            if(kmeans_cluster_weight[k])
                determinant(&covariance_matrix_sum[k][0][0], &cova_matrix_sum_determinant[k]);

        for (k = 0; k < 3; k++)
            if(cova_matrix_sum_determinant[k])
                adjoint(&covariance_matrix_sum[k][0][0], &covariance_matrix_adj_mul_weight[k][0][0], (double)kmeans_cluster_weight[k]);

        /*************************LUT Creation/Updation**********************
        ********************************************************************/



        index_weight = get_maximum_index_weight(&kmeans_cluster_weight_copy[0]);
        i = index_weight[0];

        index_weight = get_maximum_index_weight(&kmeans_cluster_weight_copy[0]);
        while (*(index_weight + 1) != 0)
        {
            i = *(index_weight + 0);

            /*****LUT Initialization*****/
            if(COLOR_MODELS == 0 && index_weight[1] > 0)
            {
                LUT[COLOR_MODELS].red = (COLOR_TYPE)kmeans[i].red;
                LUT[COLOR_MODELS].green = (COLOR_TYPE)kmeans[i].green;
                LUT[COLOR_MODELS].blue = (COLOR_TYPE)kmeans[i].blue;
                LUT[COLOR_MODELS].weight = kmeans_cluster_weight[i];

                for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                {
                    LUT[COLOR_MODELS].covariance_matrix_sum[j][k] = covariance_matrix_sum[i][j][k];
                    LUT[COLOR_MODELS].covariance_matrix_adj_mul_weight[j][k] = covariance_matrix_adj_mul_weight[i][j][k];
                }

                LUT[COLOR_MODELS].determinant = cova_matrix_sum_determinant[i];
                kmeans_cluster_weight_copy[i] = 0;
                COLOR_MODELS++;
//                fprintf(fp_log, "Kmeans[%d] added to LUT\n", i);
                index_weight = get_maximum_index_weight(&kmeans_cluster_weight_copy[0]);
                i = index_weight[0];
            }

            /*****LUT Updataion*****/
            if(kmeans_cluster_weight_copy[i])
            for (j = 0; j < COLOR_MODELS; j++)
            {
                criterion = 0;
                for (k = 0; k < 3; k++)//columns
                {
                    cova_diff[0][k] = covariance_matrix_sum[i][0][k] * LUT[j].weight + LUT[j].covariance_matrix_sum[0][k] * kmeans_cluster_weight[i];
                    cova_diff[1][k] = covariance_matrix_sum[i][1][k] * LUT[j].weight + LUT[j].covariance_matrix_sum[1][k] * kmeans_cluster_weight[i];
                    cova_diff[2][k] = covariance_matrix_sum[i][2][k] * LUT[j].weight + LUT[j].covariance_matrix_sum[2][k] * kmeans_cluster_weight[i];
                }

                adjoint(&cova_diff[0][0], &cova_diff_adj_mul_w1w2[0][0], LUT[j].weight * (double)kmeans_cluster_weight[i]);
                determinant(&cova_diff[0][0], &cova_diff_determinant);

                diff_r = (COLOR_TYPE)kmeans[i].red - (COLOR_TYPE)LUT[j].red;
                diff_g = (COLOR_TYPE)kmeans[i].green - (COLOR_TYPE)LUT[j].green;
                diff_b = (COLOR_TYPE)kmeans[i].blue - (COLOR_TYPE)LUT[j].blue;

                criterion += (COLOR_TYPE_2)diff_r * cova_diff_adj_mul_w1w2[0][0] * (COLOR_TYPE_2) diff_r;
                criterion += (COLOR_TYPE_2)diff_r * cova_diff_adj_mul_w1w2[0][1] * (COLOR_TYPE_2) diff_g;
                criterion += (COLOR_TYPE_2)diff_r * cova_diff_adj_mul_w1w2[0][2] * (COLOR_TYPE_2) diff_b;
                criterion += (COLOR_TYPE_2)diff_g * cova_diff_adj_mul_w1w2[1][0] * (COLOR_TYPE_2) diff_r;
                criterion += (COLOR_TYPE_2)diff_g * cova_diff_adj_mul_w1w2[1][1] * (COLOR_TYPE_2) diff_g;
                criterion += (COLOR_TYPE_2)diff_g * cova_diff_adj_mul_w1w2[1][2] * (COLOR_TYPE_2) diff_b;
                criterion += (COLOR_TYPE_2)diff_b * cova_diff_adj_mul_w1w2[2][0] * (COLOR_TYPE_2) diff_r;
                criterion += (COLOR_TYPE_2)diff_b * cova_diff_adj_mul_w1w2[2][1] * (COLOR_TYPE_2) diff_g;
                criterion += (COLOR_TYPE_2)diff_b * cova_diff_adj_mul_w1w2[2][2] * (COLOR_TYPE_2) diff_b;

                if (fabsl(criterion) <= (double)1 * fabsl(cova_diff_determinant))	/*****LUT updation*****/
                {
                    weight_sum = LUT[j].weight + (COLOR_TYPE_2)kmeans_cluster_weight[i];
                    LUT[j].red = (COLOR_TYPE)((COLOR_TYPE_2)(LUT[j].red * LUT[j].weight + kmeans[i].red * kmeans_cluster_weight[i])/weight_sum);
                    LUT[j].green = (COLOR_TYPE)((COLOR_TYPE_2)(LUT[j].green * LUT[j].weight + kmeans[i].green * kmeans_cluster_weight[i])/weight_sum);
                    LUT[j].blue = (COLOR_TYPE)((COLOR_TYPE_2)(LUT[j].blue * LUT[j].weight + kmeans[i].blue * kmeans_cluster_weight[i])/weight_sum);

                    LUT[j].weight = (double)LUT[j].weight + (double)kmeans_cluster_weight[i];


                    for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                        LUT[j].covariance_matrix_sum[k][l] = LUT[j].covariance_matrix_sum[k][l] + covariance_matrix_sum[i][k][l];

                    determinant(&LUT[j].covariance_matrix_sum[0][0], &LUT[j].determinant);

                    if(LUT[j].determinant)
                        adjoint(&LUT[j].covariance_matrix_sum[0][0], &LUT[j].covariance_matrix_adj_mul_weight[0][0], LUT[j].weight);
                    else
                        LUT[j].covariance_matrix_adj_mul_weight[0][0] = LUT[j].covariance_matrix_adj_mul_weight[1][1] = LUT[j].covariance_matrix_adj_mul_weight[2][2] = 0xffffffffffffffff;

                    kmeans_cluster_weight_copy[i] = 0;
//                    fprintf(fp_log, "Kmeans[%d] updated to LUT[%d]\n", i, j);
                    break;
                }
            }

            /*****LUT Addition*****/
            if(kmeans_cluster_weight_copy[i])
            if(COLOR_MODELS < MAX_LUT_SIZE && index_weight[1] > 0)
            {
                LUT[COLOR_MODELS].red = (COLOR_TYPE)kmeans[i].red;
                LUT[COLOR_MODELS].green = (COLOR_TYPE)kmeans[i].green;
                LUT[COLOR_MODELS].blue = (COLOR_TYPE)kmeans[i].blue;
                LUT[COLOR_MODELS].weight = kmeans_cluster_weight[i];

                for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                {
                    LUT[COLOR_MODELS].covariance_matrix_sum[j][k] = covariance_matrix_sum[i][j][k];
                    LUT[COLOR_MODELS].covariance_matrix_adj_mul_weight[j][k] = covariance_matrix_adj_mul_weight[i][j][k];
                }

                LUT[COLOR_MODELS].determinant = cova_matrix_sum_determinant[i];
                kmeans_cluster_weight_copy[i] = 0;
                COLOR_MODELS++;
//                fprintf(fp_log, "Kmeans[%d] added to LUT\n", i);
                index_weight = get_maximum_index_weight(&kmeans_cluster_weight_copy[0]);
                i = index_weight[0];
            }

            /*****LUT Replacement*****/
            if(kmeans_cluster_weight_copy[i])
            {
                LUT_min_index_weight = get_minimum_index_weight(LUT);
                j = LUT_min_index_weight[0];

                if (LUT_min_index_weight[1] < index_weight[1])
                {
                    LUT[j].red = kmeans[i].red;
                    LUT[j].green = kmeans[i].green;
                    LUT[j].blue = kmeans[i].blue;
                    LUT[j].weight = kmeans_cluster_weight[i];
                    for (k = 0; k < 3; k++)
                    for (l = 0; l < 3; l++)
                    {
                        LUT[j].covariance_matrix_sum[k][l] = covariance_matrix_sum[i][k][l];
                        LUT[j].covariance_matrix_adj_mul_weight[k][l] = covariance_matrix_adj_mul_weight[i][k][l];
                    }

                    LUT[COLOR_MODELS].determinant = cova_matrix_sum_determinant[i];
                    kmeans_cluster_weight_copy[i] = 0;
//                    fprintf(fp_log, "Kmeans[%d] Replaced LUT[%d]\n", i, j);
                }
            }

            if(kmeans_cluster_weight_copy[i])
            {
                kmeans_cluster_weight_copy[i] = 0;
//                fprintf(fp_log, "Kmeans[%d] discarded\n", i);
            }
            index_weight = get_maximum_index_weight(&kmeans_cluster_weight_copy[0]);
        }

/*********************************  Horizon Detection before Scoring Pixels  *****************************/
        memset(horizon_sum_vector, (unsigned short)0, sizeof(unsigned short) * ROWS);

        for(i = 0; i < ROWS; i++)
        for(j = 0; j < COLS; j++)
        {
            count_h = 0;
            if(((img.red[i][j]>>BIT_SHFIT) - mean_ROI.red)*((img.red[i][j]>>BIT_SHFIT) - mean_ROI.red) >= variance_ROI.green)
            count_h++;
            if(((img.green[i][j]>>BIT_SHFIT) - mean_ROI.red)*((img.red[i][j]>>BIT_SHFIT) - mean_ROI.red) >= variance_ROI.green)
            count_h++;
            if(((img.blue[i][j]>>BIT_SHFIT) - mean_ROI.red)*((img.red[i][j]>>BIT_SHFIT) - mean_ROI.red) >= variance_ROI.blue)
            count_h++;

            if(count_h==3)
                horizon_sum_vector[i]++; //horizon[i][j] = 1,
        }

        HORIZON = 0;
        max_horizon = horizon_sum_vector[HORIZON];
        for(i = 1; i < ROWS; i++)
        {
            if(max_horizon < horizon_sum_vector[i])
            {
                HORIZON = i;
                max_horizon = horizon_sum_vector[HORIZON];
            }
        }
/********************************************************************************************************/

/*********************************  Smoothening before Scoring  ******************************************/

        unsigned short smooth_filter[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
        #ifdef REDUCE_SCORING
            for(i = 1; i < ROWS*2-1; i++)
            for(j = 1; j < COLS*2-1; j++)
            {
                smooth_sum.red = smooth_sum.green = smooth_sum.blue = 0;
                for(k = 0; k < 3; k++)
                for(l = 0; l < 3; l++)
                {
                    smooth_sum.red += (unsigned  short)img_l1.red[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                    smooth_sum.green += (unsigned  short)img_l1.green[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                    smooth_sum.blue += (unsigned  short)img_l1.blue[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                }

    //            Divided by 16
                img_l1.red[i][j] = (unsigned char)(smooth_sum.red >> 4);
                img_l1.green[i][j] = (unsigned char)(smooth_sum.green >> 4);
                img_l1.blue[i][j] = (unsigned char)(smooth_sum.blue >> 4);
            }
        #else
            for(i = 1; i < ROWS-1; i++)
            for(j = 1; j < COLS-1; j++)
            {
                smooth_sum.red = smooth_sum.green = smooth_sum.blue = 0;
                for(k = 0; k < 3; k++)
                for(l = 0; l < 3; l++)
                {
                    smooth_sum.red += (unsigned  short)img.red[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                    smooth_sum.green += (unsigned  short)img.green[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                    smooth_sum.blue += (unsigned  short)img.blue[(i+k-1)][j+l-1] * smooth_filter[k][l];// << (2 - (i+1 % 2) - (j+1 % 2));
                }

    //            Divided by 16
                img.red[i][j] = (unsigned char)(smooth_sum.red >> 4);
                img.green[i][j] = (unsigned char)(smooth_sum.green >> 4);
                img.blue[i][j] = (unsigned char)(smooth_sum.blue >> 4);
            }
        #endif


/*********************************  Scoring Pixels for Navigable Terrain    *****************************/

        #ifdef REDUCE_SCORING
            static unsigned char navigable[ROWS*2][COLS*2];

            color_s_d_64 distance;
            color temp;
            for(i = HORIZON*2; i < ROWS*2; i++)
            for(j = 0; j < COLS*2; j++)
            {
                temp.red = img_l1.red[i][j] >> BIT_SHFIT;
                temp.green = img_l1.green[i][j] >> BIT_SHFIT;
                temp.blue = img_l1.blue[i][j] >> BIT_SHFIT;
                for(k = 0; k < COLOR_MODELS; k++)
                {
                    distance.red = distance.green = distance.blue = 0;

                    diff_r = (COLOR_TYPE_2)temp.red - (COLOR_TYPE_2)LUT[k].red;
                    diff_g = (COLOR_TYPE_2)temp.green - (COLOR_TYPE_2)LUT[k].green;
                    diff_b = (COLOR_TYPE_2)temp.blue - (COLOR_TYPE_2)LUT[k].blue;

                    if(LUT[k].determinant != 0)
                    {
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][0] * diff_r;
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][1] * diff_g;
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][2] * diff_b;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][0] * diff_r;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][1] * diff_g;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][2] * diff_b;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][0] * diff_r;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][1] * diff_g;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][2] * diff_b;

                        THRESHOLD = (double)10 * fabsl(LUT[k].determinant);
                        fabsl_distance = fabsl(distance.red) + fabsl(distance.green) + fabsl(distance.blue);
                        if(fabsl_distance < THRESHOLD)// 10 is good
                        {
                            navigable[i][j] = 1;
                            break;
                        }
                    }
                }
            }
        #else
            static unsigned char navigable[ROWS][COLS];

            color_s_d_64 distance;
            color temp;
            for(i = HORIZON; i < ROWS; i++)
            for(j = 0; j < COLS; j++)
            {
                temp.red = img.red[i][j] >> BIT_SHFIT;
                temp.green = img.green[i][j] >> BIT_SHFIT;
                temp.blue = img.blue[i][j] >> BIT_SHFIT;
                for(k = 0; k < COLOR_MODELS; k++)
                {
                    distance.red = distance.green = distance.blue = 0;

                    diff_r = (COLOR_TYPE_2)temp.red - (COLOR_TYPE_2)LUT[k].red;
                    diff_g = (COLOR_TYPE_2)temp.green - (COLOR_TYPE_2)LUT[k].green;
                    diff_b = (COLOR_TYPE_2)temp.blue - (COLOR_TYPE_2)LUT[k].blue;

                    if(LUT[k].determinant != 0)
                    {
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][0] * diff_r;
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][1] * diff_g;
                        distance.red += diff_r * LUT[k].covariance_matrix_adj_mul_weight[0][2] * diff_b;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][0] * diff_r;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][1] * diff_g;
                        distance.green += diff_g * LUT[k].covariance_matrix_adj_mul_weight[1][2] * diff_b;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][0] * diff_r;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][1] * diff_g;
                        distance.blue += diff_b * LUT[k].covariance_matrix_adj_mul_weight[2][2] * diff_b;

                        THRESHOLD = (double)10 * fabsl(LUT[k].determinant);
                        fabsl_distance = fabsl(distance.red) + fabsl(distance.green) + fabsl(distance.blue);
                        if(fabsl_distance < THRESHOLD)// 10 is good
                        {
                            navigable[i][j] = 1;
                            break;
                        }
                    }
                }
            }
        #endif
    }
    closedir(dir);
    exit(0);
    return 0;
}

unsigned short* get_maximum_index_weight(unsigned short *kmeans_cluster_weight_copy)
{
	unsigned char i = 0;
	static unsigned short index_weight[2];
	index_weight[0] = 0;
	index_weight[1] = *(kmeans_cluster_weight_copy + 0);
	for (i = 1; i < 3; i++)
	if (*(kmeans_cluster_weight_copy + i) > index_weight[1])
	{
		index_weight[1] = *(kmeans_cluster_weight_copy + i);
		index_weight[0] = i;
	}
	return index_weight;
}

unsigned short* get_minimum_index_weight(color_model* LUT)
{
	unsigned char i = 0;
	static unsigned short index_weight[2];
	index_weight[0] = 0;
	index_weight[1] = (LUT + 0)->weight;
	for (i = 0; i < COLOR_MODELS; i++)
	if ((LUT + i)->weight < index_weight[1])
	{
		index_weight[1] = (LUT + i)->weight;
		index_weight[0] = i;
	}
	return index_weight;
}

void determinant(double* covariance_matrix_sum, double* cova_matrix_sum_determinant)
{
    cova_matrix_sum_determinant[0] = 0;
    cova_matrix_sum_determinant[0] += covariance_matrix_sum[0] * ((covariance_matrix_sum[4] * covariance_matrix_sum[8]) - (covariance_matrix_sum[5] * covariance_matrix_sum[7]));
    cova_matrix_sum_determinant[0] += covariance_matrix_sum[1] * (double)-1 * ((covariance_matrix_sum[3] * covariance_matrix_sum[8]) - (covariance_matrix_sum[5] * covariance_matrix_sum[6]));
    cova_matrix_sum_determinant[0] += covariance_matrix_sum[2] * ((covariance_matrix_sum[3] * covariance_matrix_sum[7]) - (covariance_matrix_sum[4] * covariance_matrix_sum[6]));
}

void adjoint(double* covariance_matrix_sum, double* covariance_matrix_adj_mul_model_weight, double model_weight)
{
    covariance_matrix_adj_mul_model_weight[0] = model_weight * ((covariance_matrix_sum[4] * covariance_matrix_sum[8]) - (covariance_matrix_sum[5] * covariance_matrix_sum[7]));
    covariance_matrix_adj_mul_model_weight[1] = model_weight * (double)-1 * ((covariance_matrix_sum[3] * covariance_matrix_sum[8]) - (covariance_matrix_sum[5] * covariance_matrix_sum[6]));
    covariance_matrix_adj_mul_model_weight[2] = model_weight * ((covariance_matrix_sum[3] * covariance_matrix_sum[7]) - (covariance_matrix_sum[4] * covariance_matrix_sum[6]));
    covariance_matrix_adj_mul_model_weight[3] = covariance_matrix_adj_mul_model_weight[1];
    covariance_matrix_adj_mul_model_weight[4] = model_weight* ((covariance_matrix_sum[0]) * (covariance_matrix_sum[8]) - (covariance_matrix_sum[2]) * (covariance_matrix_sum[6]));
    covariance_matrix_adj_mul_model_weight[5] = model_weight * (double)-1 * ((covariance_matrix_sum[0]) * (covariance_matrix_sum[7]) - (covariance_matrix_sum[1]) * (covariance_matrix_sum[6]));
    covariance_matrix_adj_mul_model_weight[6] = covariance_matrix_adj_mul_model_weight[2];
    covariance_matrix_adj_mul_model_weight[7] = covariance_matrix_adj_mul_model_weight[5];
    covariance_matrix_adj_mul_model_weight[8] = model_weight * ((covariance_matrix_sum[0]) * (covariance_matrix_sum[4]) - (covariance_matrix_sum[1]) * (covariance_matrix_sum[3]));
}