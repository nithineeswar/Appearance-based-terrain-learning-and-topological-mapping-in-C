#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#define BIT_SHFIT 4
#define COMPARE ==
#define MALAGA DATABASE

//#define MAP_BUILDING
#define LOCALIZATION


#define HIST_SIZE 1000

#ifdef IDOL2
    #define IMG_SIZE 76800
    #define BLOCK_SIZE 80
    #define ROW_BLOCKS 3
    #define COLUMN_BLOCKS 4
    #define THRESHOLD 164999

    char DIRNAME[4][51] =   {
                                {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_1\\binary\\"},
                                {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_2\\binary\\"},
                                {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_3\\binary\\"},
                                {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_4\\binary\\"}
                            };

    char FILENAME_ODOM[4][46] = {
                                    {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_1.odom"},
                                    {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_2.odom"},
                                    {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_3.odom"},
                                    {"E:\\Datasets\\IDOL2\\IDOL2_dummy_sunny_4.odom"}
                                };

    unsigned int IMAGE_NUMBERS[4] = {894, 909, 950, 999};

    typedef struct
    {
        unsigned char red[240][320];
        unsigned char green[240][320];
        unsigned char blue[240][320];
    } image;

    #define DIRNUM 3
    #define FILENAME_NODE "E:\\Datasets\\IDOL2\\Results\\IDOL2_DB_3_topo_dis_g_plus_l_greater_than_164999.node"
    #ifdef LOCALIZATION
        #define NODES 167
        #define FILENAME_LOG "E:\\Datasets\\IDOL2\\Results\\IDOL2_DB_4_topo_localization_log_DB_3.txt"
    #elif defined MAP_BUILDING
        #define FILENAME_POSE_LOG "E:\\Datasets\\IDOL2\\Results\\IDOL2_DB_3_topological_nodes_pose.txt"
        #define FILENAME_LOG "E:\\Datasets\\IDOL2\\Results\\IDOL2_DB_3_topo_map_building_log.txt"
    #endif
#elif defined MALAGA
    #define THRESHOLD 49999
    #define DIRNUM 0
    char DIRNAME[1][76] = {"E:\\Datasets\\Malaga_Parking_6L\\selcted_reduced_binary_images_camera_left\\"};
    char FILENAME_ODOM[1][45] = {"E:\\Datasets\\Malaga_Parking_6L\\MALAGA.odom"};
    unsigned int IMAGE_NUMBERS[1] = {3072};
    #define IMG_SIZE 49152
    #define BLOCK_SIZE 64
    #define ROW_BLOCKS 3
    #define COLUMN_BLOCKS 4

    typedef struct
    {
        unsigned char red[192][256];
        unsigned char green[192][256];
        unsigned char blue[192][256];
    } image;

    #define FILENAME_NODE "E:\\Datasets\\Malaga_Parking_6L\\malaga_selected_49999.node"

    #ifdef LOCALIZATION
        #define NODES 225
        #define FILENAME_LOG "E:\\Datasets\\Malaga_Parking_6L\\malaga_selected_localization_log_49999.txt"
    #elif defined MAP_BUILDING
        #define FILENAME_POSE_LOG "E:\\Datasets\\Malaga_Parking_6L\\malaga_selected_topological_nodes_pose_49999.txt"
        #define FILENAME_LOG "E:\\Datasets\\Malaga_Parking_6L\\malaga_selected_map_building_log_49999.txt"
    #endif
#endif

typedef struct
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
}color;

typedef struct
{
    unsigned int red;
    unsigned int green;
    unsigned int blue;
}color_32;

typedef struct
{
    float x;
    float y;
    float theta;
}coordinate;

typedef struct
{
    coordinate pose;
    unsigned char block_histogram[ROW_BLOCKS][COLUMN_BLOCKS][HIST_SIZE][3];
    unsigned short global_histogram[16][16][16];
    unsigned short block_histogram_weight[ROW_BLOCKS][COLUMN_BLOCKS][HIST_SIZE];
    unsigned char block_bin_presence[ROW_BLOCKS][COLUMN_BLOCKS][4096];
    color mean[ROW_BLOCKS][COLUMN_BLOCKS];
    unsigned short pixels_at_mean[ROW_BLOCKS][COLUMN_BLOCKS];
    unsigned short block_connectivity_score[ROW_BLOCKS][COLUMN_BLOCKS];
}node;

unsigned int *g_dis, *l_dis, *m_dis, *n_m_dis, *con_dis;

void get_curnt_image(image*, struct dirent*, unsigned char );
void compute_features(node* , image*);
void compute_range(unsigned char, unsigned char, unsigned char, unsigned char*, unsigned char*, unsigned char*);
void compute_index(unsigned char, unsigned char, unsigned char*, unsigned char*, unsigned char*, unsigned char*);
void compute_distance(node*, node*, unsigned int*, unsigned short);
unsigned int* get_min(unsigned int*, unsigned int);

int main()
{
    float *x, *y, *theta;
    unsigned short nodes_count, image_number;
    node *saved, current;
    unsigned int *distance, *index_min_distance;
    image L2;
    FILE *fp, *fp_log;
    struct dirent *dptr;
    DIR *dir;

    fp_log = fopen(FILENAME_LOG,"w");
    if (fp_log == NULL)
        perror("File opening error"), exit(-1);

    fp = fopen(FILENAME_ODOM[DIRNUM], "rb");
    if (fp == NULL)
        perror("File opening error"), exit(-1);

    x = (float*)malloc(sizeof(float) * IMAGE_NUMBERS[DIRNUM]);
    y = (float*)malloc(sizeof(float) * IMAGE_NUMBERS[DIRNUM]);
    theta = (float*)malloc(sizeof(float) * IMAGE_NUMBERS[DIRNUM]);

    fread(x, sizeof(float), IMAGE_NUMBERS[DIRNUM], fp);
    fread(y, sizeof(float), IMAGE_NUMBERS[DIRNUM], fp);
    fread(theta, sizeof(float), IMAGE_NUMBERS[DIRNUM], fp);
    fclose(fp);

    dir = opendir(DIRNAME[DIRNUM]);
    if(dir == NULL)
        perror("Directory opening error"), exit(-1);
    dptr=readdir(dir), dptr=readdir(dir);

#ifdef LOCALIZATION

    nodes_count = NODES;
    saved = (node*)malloc(sizeof(node) * nodes_count);
    distance = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
    g_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
    l_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
    m_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
    n_m_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
    con_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));

    fp = fopen(FILENAME_NODE, "rb");
    fread(saved, sizeof(node), nodes_count, fp);
    fclose(fp);

    image_number = 0;
    while((dptr = readdir(dir)))
    {
        image_number++;
        get_curnt_image(&L2, dptr, DIRNUM);

        current.pose.x = x[image_number-1];
        current.pose.y = y[image_number-1];
        current.pose.theta = theta[image_number-1];

        compute_features(&current, &L2);

        if(image_number == 409)
            puts("");

        printf("%d ", image_number);
        compute_distance(saved, &current, distance, nodes_count);
        index_min_distance = get_min(distance, nodes_count);
        fprintf(fp_log, "%3.6f %3.6f %3.6f\t%2d %6d\t %3.6f %3.6f %3.6f\n", current.pose.x, current.pose.y, current.pose.theta, index_min_distance[0], index_min_distance[1], saved[index_min_distance[0]].pose.x, saved[index_min_distance[0]].pose.y, saved[index_min_distance[0]].pose.theta);
        printf("localized to %d\t%d\n", index_min_distance[0], index_min_distance[1]);
        memset(distance, (unsigned int)0, sizeof(unsigned int) * nodes_count);
    }

#elif defined MAP_BUILDING

    int i, diff_theta;
    unsigned int accumulated_theta = 0;
    float previous_theta;
    fp = fopen(FILENAME_POSE_LOG, "w");
    if (fp == NULL)
        perror("File opening error"), exit(-1);

    nodes_count = 0;
    image_number = 0;
    while((dptr = readdir(dir)))
    {
        image_number++;
        get_curnt_image(&L2, dptr, DIRNUM);

        current.pose.x = x[image_number-1];
        current.pose.y = y[image_number-1];
        current.pose.theta = theta[image_number-1];

        compute_features(&current, &L2);

/************************BUILD TOPOLOGICAL MAP*************************/

       if(!nodes_count)
        {
            nodes_count++;
            saved = (node*)malloc(sizeof(node));

            saved[nodes_count - 1] = current;
            fprintf(fp, "%3.10f %3.10f %3.10f\n", saved[nodes_count-1].pose.x, saved[nodes_count-1].pose.y, saved[nodes_count-1].pose.theta);

            distance = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
            g_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
            l_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
            m_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
            n_m_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
            con_dis = (unsigned int*)calloc(nodes_count, sizeof(unsigned int));
        }
        else
        {
            compute_distance(saved, &current,distance, nodes_count);
            index_min_distance = get_min(distance, nodes_count);
            i = index_min_distance[0];

            diff_theta = abs(current.pose.theta - previous_theta) < 360 - abs(current.pose.theta - previous_theta) ? abs(current.pose.theta - previous_theta) : 360 - abs(current.pose.theta - previous_theta);
            accumulated_theta += diff_theta;

            printf("%u\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%u", image_number, i+1, index_min_distance[1], g_dis[i], l_dis[i], m_dis[i], n_m_dis[i], con_dis[i], accumulated_theta);
            if(distance[i] > THRESHOLD || (int)accumulated_theta >= 20)
            {
                accumulated_theta = 0;
                nodes_count++;
                saved = (node*)realloc(saved, nodes_count * sizeof(node));

                saved[nodes_count - 1] = current;
                fprintf(fp, "%3.10f %3.10f %3.10f\n", saved[nodes_count-1].pose.x, saved[nodes_count-1].pose.y, saved[nodes_count-1].pose.theta);

                distance = (unsigned int*)realloc(distance, nodes_count * sizeof(unsigned int));
                g_dis = (unsigned int*)realloc(g_dis, nodes_count * sizeof(unsigned int));
                l_dis = (unsigned int*)realloc(l_dis, nodes_count * sizeof(unsigned int));
                m_dis = (unsigned int*)realloc(m_dis, nodes_count * sizeof(unsigned int));
                n_m_dis = (unsigned int*)realloc(n_m_dis, nodes_count * sizeof(unsigned int));
                con_dis = (unsigned int*)realloc(con_dis, nodes_count * sizeof(unsigned int));

                memset(distance, (unsigned int)0, sizeof(unsigned int) * nodes_count);
                memset(g_dis, (unsigned int)0, sizeof(unsigned int) * nodes_count);
                memset(l_dis, (unsigned int)0, sizeof(unsigned int) * nodes_count);
                memset(m_dis, (unsigned int)0, sizeof(unsigned int) * nodes_count);
                memset(n_m_dis, (unsigned int)0, sizeof(unsigned int) * nodes_count);
                memset(con_dis, (unsigned int)0, sizeof(unsigned int) * nodes_count);

                i = nodes_count - 1;
            }
            printf("\t%u\n", nodes_count);

            fprintf(fp_log, "%3d %3d %3d %6d %3.6f %3.6f %3.6f\n", nodes_count, image_number, i, index_min_distance[1], current.pose.x, current.pose.y, current.pose.theta);
        }
        previous_theta = current.pose.theta;
    }

    fp = fopen(FILENAME_NODE, "wb");
    fwrite(saved, sizeof(node), nodes_count, fp);
    fclose(fp);

#endif
    free(x);
    free(y);
    free(theta);
    closedir(dir);
    fclose(fp_log);
    return 0;
}

void get_curnt_image(image* L2, struct dirent* dptr, unsigned char directory_index)
{
    FILE *fp;
    unsigned char RGB_arr[IMG_SIZE * 3];
    char* str;

    str = calloc(sizeof(DIRNAME[directory_index]) + dptr->d_namlen, sizeof(char));
    strcpy(str, DIRNAME[directory_index]);
    strcat(str, dptr->d_name);

    fp = fopen(str, "rb");
    if (fp == NULL)
        perror("File opening error"), exit(-1);
    fread(RGB_arr, sizeof(unsigned char), (IMG_SIZE * 3), fp);
    fclose(fp);

    memcpy(L2->red, RGB_arr, IMG_SIZE);
    memcpy(L2->green, (RGB_arr + IMG_SIZE), IMG_SIZE);
    memcpy(L2->blue, (RGB_arr + 2 * IMG_SIZE), IMG_SIZE);
}

void compute_features(node* current, image *L2)
{
	int index_one, index_two, i, j, k, block_row, block_column;
	unsigned char red, green, blue;

	/*******************Local and Global Image Statistics*******************/

	unsigned char red_range[3], green_range[3], blue_range[3];
	unsigned char row_start, row_end, col_start, col_end;
	color_32 mean_sum;

	memset(current->block_histogram, (unsigned char)0, sizeof(unsigned char) * ROW_BLOCKS * COLUMN_BLOCKS * HIST_SIZE * 3);
	memset(current->block_histogram_weight, (unsigned short)0, sizeof(unsigned short) * ROW_BLOCKS * COLUMN_BLOCKS * HIST_SIZE);
	memset(current->global_histogram, (unsigned short)0, sizeof(unsigned short)* 16 * 16 * 16);
	memset(current->block_bin_presence, (char)0, sizeof(char)* ROW_BLOCKS * COLUMN_BLOCKS * 4096);

	for (block_row = 0; block_row < ROW_BLOCKS; block_row++)
	for (block_column = 0; block_column < COLUMN_BLOCKS; block_column++)
	{
		mean_sum.red = mean_sum.green = mean_sum.blue = 0;//     *****

		for (i = 0; i < BLOCK_SIZE; i++)
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			index_one = ((block_row * BLOCK_SIZE) + i);
			index_two = ((block_column * BLOCK_SIZE) + j);

			red = L2->red[index_one][index_two] >> BIT_SHFIT;
			green = L2->green[index_one][index_two] >> BIT_SHFIT;
			blue = L2->blue[index_one][index_two] >> BIT_SHFIT;

			for (k = 0; k < HIST_SIZE; k++)
			{
				if (current->block_histogram[block_row][block_column][k][0] COMPARE red && current->block_histogram[block_row][block_column][k][1] COMPARE green && current->block_histogram[block_row][block_column][k][2] COMPARE blue)
				{

					current->block_histogram_weight[block_row][block_column][k] += 1;
					break;
				}
				if ((current->block_histogram[block_row][block_column][k][0] + current->block_histogram[block_row][block_column][k][1] + current->block_histogram[block_row][block_column][k][2] + current->block_histogram_weight[block_row][block_column][k]) COMPARE 0)
				{
					current->block_histogram[block_row][block_column][k][0] = red;
					current->block_histogram[block_row][block_column][k][1] = green;
					current->block_histogram[block_row][block_column][k][2] = blue;
					current->block_histogram_weight[block_row][block_column][k] += 1;
					break;
				}
			}

			current->block_bin_presence[block_row][block_column][red * 256 + green * 16 + blue] = 1;

			mean_sum.red += red;
			mean_sum.green += green;
			mean_sum.blue += blue;

			/*****Global Histogram*****/
			current->global_histogram[red][green][blue] += 1;
		}

		/*****Block mean*****/
		current->mean[block_row][block_column].red = (unsigned char)floor((double)mean_sum.red / (double)6400 + 0.5);
		current->mean[block_row][block_column].green = (unsigned char)floor((double)mean_sum.green / (double)6400 + 0.5);
		current->mean[block_row][block_column].blue = (unsigned char)floor((double)mean_sum.blue / (double)6400 + 0.5);

	}

	memset(current->pixels_at_mean, (unsigned short)0, sizeof(unsigned short)* ROW_BLOCKS * COLUMN_BLOCKS);
	memset(current->block_connectivity_score, (unsigned short)0, sizeof(unsigned short)* ROW_BLOCKS * COLUMN_BLOCKS);

	for (block_row = 0; block_row < ROW_BLOCKS; block_row++)
	for (block_column = 0; block_column < COLUMN_BLOCKS; block_column++)
	{//      *****BLOCK LOOP
		/*****Number of pixels at mean with +&- 1 neighbours*****/

		red = current->mean[block_row][block_column].red;
		green = current->mean[block_row][block_column].green;
		blue = current->mean[block_row][block_column].blue;

		compute_range(red, green, blue, red_range, green_range, blue_range);

		for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
		for (k = 0; k < 3; k++)
		if (current->block_bin_presence[(red_range[i] * 256) + (green_range[j] * 16) + blue_range[k]])
		for (index_one = 0; index_one < HIST_SIZE && current->block_histogram_weight[index_one]; index_one++)
		if (current->block_histogram[block_row][block_column][index_one][0] COMPARE red_range[i] && current->block_histogram[block_row][block_column][index_one][1] COMPARE green_range[j] && current->block_histogram[block_row][block_column][index_one][2] COMPARE blue_range[k])
		{
			current->pixels_at_mean[block_row][block_column] += current->block_histogram_weight[block_row][block_column][index_one];
			break;
		}

		/************Block connectivity scores************/

		compute_index(block_row, block_column, &row_start, &row_end, &col_start, &col_end);

		for (i = row_start; i <= row_end; i++)
		for (j = col_start; j <= col_end; j++)
		{
			current->block_connectivity_score[block_row][block_column] += abs(current->mean[block_row][block_column].red - current->mean[i][j].red);
			current->block_connectivity_score[block_row][block_column] += abs(current->mean[block_row][block_column].green - current->mean[i][j].green);
			current->block_connectivity_score[block_row][block_column] += abs(current->mean[block_row][block_column].blue - current->mean[i][j].blue);
		}
	}
}

void compute_range(unsigned char red, unsigned char green, unsigned char blue, unsigned char *red_range, unsigned char *green_range, unsigned char *blue_range)
{
	unsigned char i, weight;
	if (red < 2)
	for (i = 0, weight = 0; i < 3; i++, weight++)
		red_range[i] = weight;

	if (green < 2)
	for (i = 0, weight = 0; i < 3; i++, weight++)
		green_range[i] = weight;

	if (blue < 2)
	for (i = 0, weight = 0; i < 3; i++, weight++)
		blue_range[i] = weight;

	if (red > 1 && red < 13)
	for (i = 0, weight = (red - 1); i < 3; i++, weight++)
		red_range[i] = weight;

	if (green > 1 && green < 13)
	for (i = 0, weight = (green - 1); i < 3; i++, weight++)
		green_range[i] = weight;

	if (blue > 1 && blue < 13)
	for (i = 0, weight = (blue - 1); i < 3; i++, weight++)
		blue_range[i] = weight;

	if (red > 12)
	for (i = 0, weight = 13; i < 3; i++, weight++)
		red_range[i] = weight;

	if (green > 12)
	for (i = 0, weight = 13; i < 3; i++, weight++)
		green_range[i] = weight;

	if (blue > 12)
	for (i = 0, weight = 13; i < 3; i++, weight++)
		blue_range[i] = weight;
}

void compute_index(unsigned char row, unsigned char column, unsigned char *row_start, unsigned char *row_end, unsigned char *col_start, unsigned char *col_end)
{
	if (row > 0 && row < (ROW_BLOCKS - 1) && column > 0 && column < (COLUMN_BLOCKS - 1))//   *****   Center (Most true)
	{
		*row_start = row - 1;
		*row_end = row + 1;
		*col_start = column - 1;
		*col_end = column + 1;
	}

	else if (row COMPARE 0 && column > 0 && column < (COLUMN_BLOCKS - 1))//   *****   Upper edge
	{
		*row_start = 0;
		*row_end = 1;
		*col_start = column - 1;
		*col_end = column + 1;
	}
	else if (row > 0 && row < (ROW_BLOCKS - 1) && column COMPARE 0)//   *****   Left edge
	{
		*row_start = row - 1;
		*row_end = row + 1;
		*col_start = 0;
		*col_end = 1;
	}
	else if (row COMPARE(ROW_BLOCKS - 1) && column > 0 && column < (COLUMN_BLOCKS - 1))//   *****   lower edge
	{
		*row_start = ROW_BLOCKS - 2;
		*row_end = ROW_BLOCKS - 1;
		*col_start = column - 1;
		*col_end = column + 1;
	}
	else if (row > 0 && row < (ROW_BLOCKS - 1) && column COMPARE(COLUMN_BLOCKS - 1))//   *****   right edge
	{
		*row_start = row - 1;
		*row_end = row + 1;
		*col_start = COLUMN_BLOCKS - 2;
		*col_end = COLUMN_BLOCKS - 1;
	}
	else if (row COMPARE 0 && column COMPARE (COLUMN_BLOCKS - 1))//   *****   Upper right corner
	{
		*row_start = 0;
		*row_end = 1;
		*col_start = COLUMN_BLOCKS - 2;
		*col_end = COLUMN_BLOCKS - 1;
	}
	else if (row COMPARE 0 && column COMPARE 0)//   *****   Upper left corner
	{
		*row_start = *col_start = 0;
		*row_end = *col_end = 1;
	}
	else if (row COMPARE(ROW_BLOCKS - 1) && column COMPARE 0)//   *****   lower left corner
	{
		*row_start = ROW_BLOCKS - 2;
		*row_end = ROW_BLOCKS - 1;
		*col_start = 0;
		*col_end = 1;
	}
	else if (row COMPARE(ROW_BLOCKS - 1) && column COMPARE (COLUMN_BLOCKS - 1))//   *****   lower right corner
	{
		*row_start = ROW_BLOCKS - 2;
		*row_end = ROW_BLOCKS - 1;
		*col_start = COLUMN_BLOCKS - 2;
		*col_end = COLUMN_BLOCKS - 1;
	}
}

void compute_distance(node* saved, node* current, unsigned int *distance, unsigned short nodes_count)
{
	int hist_index, i, j, k, block_row, block_column, index_one;
	unsigned char red, green, blue;
	float diff_theta;

	for (i = 0; i < nodes_count; i++)
	{
        diff_theta = abs(current->pose.theta - saved[i].pose.theta) < 360 - abs(current->pose.theta - saved[i].pose.theta) ? abs(current->pose.theta - saved[i].pose.theta) : 360 - abs(current->pose.theta - saved[i].pose.theta);
	    if((int)diff_theta <= 30)
	    {
	        g_dis[i] = l_dis[i] = m_dis[i] = n_m_dis[i] = con_dis[i] = 0; //Initializatioon
            for (index_one = 0; index_one < 16; index_one++)
            for (j = 0; j < 16; j++)
            for (k = 0; k < 16; k++)
                g_dis[i] += abs(saved[i].global_histogram[index_one][j][k] - current->global_histogram[index_one][j][k]);

            unsigned short saved_block_hist_weight_sum;

            for (block_row = 0; block_row < ROW_BLOCKS; block_row++)
            for (block_column = 0; block_column < COLUMN_BLOCKS; block_column++)
            {
                saved_block_hist_weight_sum = 0;
                for (k = 0; k < HIST_SIZE; k++)
                {
                    if (saved[i].block_histogram_weight[block_row][block_column][k])
                        saved_block_hist_weight_sum += saved[i].block_histogram_weight[block_row][block_column][k];
                    else
                        break;
                }
                for (k = 0; k < HIST_SIZE; k++)
                {
                    if (current->block_histogram_weight[block_row][block_column][k])
                    {
                        red = current->block_histogram[block_row][block_column][k][0];
                        green = current->block_histogram[block_row][block_column][k][1];
                        blue = current->block_histogram[block_row][block_column][k][2];

                        if (saved[i].block_bin_presence[block_row][block_column][red * 256 + green * 16 + blue])
                        {
                            for (hist_index = 0; hist_index < HIST_SIZE; hist_index++)
                            if (saved[i].block_histogram[block_row][block_column][hist_index][0] COMPARE red && saved[i].block_histogram[block_row][block_column][hist_index][1] COMPARE green && saved[i].block_histogram[block_row][block_column][hist_index][2] COMPARE blue)
                            {
                                l_dis[i] += abs(saved[i].block_histogram_weight[block_row][block_column][hist_index] - current->block_histogram_weight[block_row][block_column][k]);
                                saved_block_hist_weight_sum -= saved[i].block_histogram_weight[block_row][block_column][hist_index];
                                break;
                            }
                        }
                        else
                        {
                            l_dis[i] += current->block_histogram_weight[block_row][block_column][k];
                        }
                    }
                }
                l_dis[i] += saved_block_hist_weight_sum;

                m_dis[i] += abs(saved[i].mean[block_row][block_column].red - current->mean[block_row][block_column].red) + abs(saved[i].mean[block_row][block_column].green - current->mean[block_row][block_column].green) + abs(saved[i].mean[block_row][block_column].blue - current->mean[block_row][block_column].blue);
                n_m_dis[i] += abs(saved[i].pixels_at_mean[block_row][block_column] - current->pixels_at_mean[block_row][block_column]);
                con_dis[i] += abs(saved[i].block_connectivity_score[block_row][block_column] - current->block_connectivity_score[block_row][block_column]);
            }
            #ifdef IDOL2
                distance[i] = g_dis[i] + l_dis[i];/
            #elif defined MALAGA
                distance[i] = g_dis[i] + l_dis[i] + m_dis[i] + n_m_dis[i] + con_dis[i];
            #endif
	    }
	    else
            distance[i] = g_dis[i] = l_dis[i] = m_dis[i] = n_m_dis[i] = con_dis[i] = 0xffffffff;
	}
}

unsigned int* get_min(unsigned int *distance, unsigned int nodes_count)
{
	static unsigned int index_minimum_distance[2];
	unsigned int i;
	index_minimum_distance[1] = distance[0];
	index_minimum_distance[0] = 0;

	for (i = 1; i < nodes_count; i++)

    if (index_minimum_distance[1] > distance[i])
    {
        index_minimum_distance[0] = i;
        index_minimum_distance[1] = distance[i];
    }

	return index_minimum_distance;
}
