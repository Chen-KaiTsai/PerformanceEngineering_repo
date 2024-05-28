#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mysql/mysql.h>

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

    char ***sorted_data = (char ***)malloc(num_rows * sizeof(char **)); //bufffer for the final result
    for (i = 0; i < num_rows; i++)
        sorted_data[i] = (char **)malloc(num_fields * sizeof(char *));

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

            sorted_data[row_loop][i] = (char *)malloc((lengths[i] + 1) * sizeof(char));
            strcpy(sorted_data[row_loop][i], "-1");

            if (lengths[i] == 0)
                strcpy(input_data[row_loop][i], "");
            else
                strcpy(input_data[row_loop][i], row[i]);
        }
    }

    int NTHREAD;
    scanf("%d", &NTHREAD);

    printf("num_of_thread =%d ", NTHREAD);

    //Rank sort begin
    const int N = num_rows;
    int rank, field_index, compare_result;

    double time_x, time_y;

    time_x = omp_get_wtime();

    #pragma omp parallel for private(rank, j, field_index, compare_result) num_threads(NTHREAD) schedule(guided)
    for (i = 0; i < N; i++)
    {
        rank = 0;
        for (j = 0; j < N; j++)
        {
            for (field_index = 0; field_index < num_fields; field_index++)
            {
                compare_result = strcmp(input_data[i][field_index], input_data[j][field_index]);
                if (compare_result != 0)
                {
                    if (compare_result > 0)
                        rank++;
                    break;
                }
            }
        }
        for (;;)
        {
            bool flag = false;
            for (field_index = 0; field_index < num_fields; field_index++)
            {
                if (strcmp(sorted_data[rank][field_index], input_data[i][field_index]) != 0)
                {
                    flag = true;
                    break;
                }
            }
            if (flag)
            {
                for (field_index = 0; field_index < num_fields; field_index++)
                    strcpy(sorted_data[rank][field_index], input_data[i][field_index]);
                break;
            }
            else
                rank++;
        }
    }

    time_y = omp_get_wtime();

    printf("execute time = %lf\n", time_y - time_x);

    mysql_free_result(result);
    mysql_close(con);
}