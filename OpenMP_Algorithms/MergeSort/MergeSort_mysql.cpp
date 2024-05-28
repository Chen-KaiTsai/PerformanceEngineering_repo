#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mysql/mysql.h>

int num_fields; 

void merge(char ***input_data, int i1, int j1, int i2, int j2)
{
    int i, j, k;
    i = i1;
    j = i2;
    k = 0;

    int temp_size = j2 - i1 + 1;
    char ***temp = (char ***)malloc(temp_size * sizeof(char **));

    while (i <= j1 && j <= j2)
    {
        int field_index, compare_result;
        bool flag;

        for (field_index = 0; field_index < num_fields; field_index++)
        {
            compare_result = strcmp(input_data[i][field_index], input_data[j][field_index]);
            if (compare_result < 0)
            {
                temp[k++] = input_data[i++];
                break;
            }
            else// if (compare_result > 0)
            {
                temp[k++] = input_data[j++];
                break;
            }
        }
    }

    while (i <= j1)
        temp[k++] = input_data[i++];

    while (j <= j2)
        temp[k++] = input_data[j++];

    for (i = i1, j = 0; i <= j2; i++, j++)
        input_data[i] = temp[j];

    free(temp);
}

void serial_mergesort(char ***input_data, int i, int j)
{
    int mid;

    if (i < j)
    {
        mid = (i + j) / 2;

        serial_mergesort(input_data, i, mid);
        serial_mergesort(input_data, mid + 1, j);

        merge(input_data, i, mid, mid + 1, j);
    }
}

void mergesort(char ***input_data, int i, int j, int threads)
{
    int mid;
    if (threads == 1)
        serial_mergesort(input_data, i, j);
    else if (i < j)
    {
        mid = (i + j) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergesort(input_data, i, mid, threads / 2);
            }

            #pragma omp section
            {
                mergesort(input_data, mid + 1, j, threads / 2);
            }
        }
        merge(input_data, i, mid, mid + 1, j);
    }
}

int main()
{
    char SQL_query[1024] = "SELECT LicenseType, Color, Breed, DogName FROM dog_license LIMIT 262144";
    printf("%s\n", SQL_query);

    MYSQL *con = mysql_init(NULL);

    if (con == NULL)
    {
        fprintf(stderr, "%s\n", mysql_error(con));
        return 1;
    }

    if (mysql_real_connect(con, "localhost", "username", "***", "imports", 0, NULL, 0) == NULL)
    {
        fprintf(stderr, "%s\n", mysql_error(con));
        mysql_close(con);
        return 1;
    }

    if (mysql_query(con, SQL_query)) //user enter the SQL query
    {
        fprintf(stderr, "%s\n", mysql_error(con));
        mysql_close(con);
        return 1;
    }

    MYSQL_RES *result = mysql_store_result(con);

    if (result == NULL)
    {
        fprintf(stderr, "%s\n", mysql_error(con));
        mysql_close(con);
        return 1;
    }

    num_fields = mysql_num_fields(result);
    const int num_rows = mysql_num_rows(result);
    //printf("num_fields = %d\n", num_fields);
    //printf("num_rows = %d\n", num_rows);

    int i, j, field_type_record[128]; //Now only support char*(string) planning to support int
    for (i = 0; i < num_fields; i++)
    {
        MYSQL_FIELD *field_info = mysql_fetch_field(result);
        if (IS_NUM(field_info->type))
            field_type_record[i] = 1;
        else
            field_type_record[i] = 0;
    }

    char ***input_data = (char ***)malloc(num_rows * sizeof(char **));
    for (i = 0; i < num_rows; i++)
        input_data[i] = (char **)malloc(num_fields * sizeof(char *));

    MYSQL_ROW row;
    unsigned long *lengths;

    int row_loop;

    for (row_loop = 0; row_loop < num_rows; row_loop++)
    {
        row = mysql_fetch_row(result);
        lengths = mysql_fetch_lengths(result);
        for (i = 0; i < num_fields; i++)
        {
            input_data[row_loop][i] = (char *)malloc((lengths[i] + 1) * sizeof(char));

            if (lengths[i] == 0)
                strcpy(input_data[row_loop][i], "");
            else
                strcpy(input_data[row_loop][i], row[i]);
        }
    }

    int NTHREAD;
    scanf("%d", &NTHREAD);

    printf("num_of_threads =%d \n", NTHREAD);

    int N = num_rows;

    omp_set_num_threads(NTHREAD);
    omp_set_dynamic(0);
    omp_set_nested(1);

    double time_x, time_y;

    time_x = omp_get_wtime();

    mergesort(input_data, 0, N - 1, NTHREAD);

    time_y = omp_get_wtime();

    /*
    for (i = 0; i < num_rows; i++)
    {
        for (j = 0; j < num_fields; j++)
            printf("%s \t", input_data[i][j]);
        printf("\n");
    }
    */

    printf("execute time = %lf\n", time_y - time_x);

    mysql_free_result(result);
    mysql_close(con);
}