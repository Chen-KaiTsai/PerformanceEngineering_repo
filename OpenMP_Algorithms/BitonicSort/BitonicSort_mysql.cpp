#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mysql/mysql.h>

void bitonicMerge(int start_index, int end_index, int dir, char ***input_data, int num_fields)
{
    if (dir == 1)
    {
        int num_elements = end_index - start_index + 1, field_index, compare_result;
        char **temp;

        for (int j = num_elements / 2; j > 0; j = j / 2)
        {
            for (int i = start_index; i + j <= end_index; i++)
            {
                /*
                if (input_data[i] > input_data[i + j])
                {
                    temp = input_data[i + j];
                    input_data[i + j] = input_data[i];
                    input_data[i] = temp;
                }
                */

                for (field_index = 0; field_index < num_fields; field_index++)
                {
                    compare_result = strcmp(input_data[i][field_index], input_data[i + j][field_index]);
                    if (compare_result > 0)
                    {
                        //printf("%d: %s and %d: %s\n", i, input_data[i][field_index], j + i,  input_data[i + j][field_index]);
                        temp = input_data[i + j];
                        input_data[i + j] = input_data[i];
                        input_data[i] = temp;
                        break;
                    }
                    else if(compare_result < 0)
                    {
                        break;
                    }
                }
            }
        }
    }
    else
    {
        int num_elements = end_index - start_index + 1, field_index, compare_result;
        char **temp;

        for (int j = num_elements / 2; j > 0; j = j / 2)
        {
            for (int i = start_index; i + j <= end_index; i++)
            {
                /*
                if (input_data[i + j] > input_data[i])
                {
                    temp = input_data[i + j];
                    input_data[i + j] = input_data[i];
                    input_data[i] = temp;
                }
                */
                for (field_index = 0; field_index < num_fields; field_index++)
                {
                    compare_result = strcmp(input_data[i][field_index], input_data[i + j][field_index]);
                    if (compare_result < 0)
                    {
                        //printf("%d: %s and %d: %s\n", i, input_data[i][field_index], j + i,  input_data[i + j][field_index]);
                        temp = input_data[i + j];
                        input_data[i + j] = input_data[i];
                        input_data[i] = temp;
                        break;
                    }
                    else if(compare_result > 0)
                    {
                        break;
                    }
                }
            }
        }
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

    const int num_fields = mysql_num_fields(result);
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

    double time_x, time_y;

    time_x = omp_get_wtime();

    for (int j = 2; j <= N; j *= 2)
    {
        #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
        for (int i = 0; i < N; i += j)
        {
            if (((i / j) % 2) == 0)
                bitonicMerge(i, i + j - 1, 1, input_data, num_fields);
            else
                bitonicMerge(i, i + j - 1, 0, input_data, num_fields);
        }
    }

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