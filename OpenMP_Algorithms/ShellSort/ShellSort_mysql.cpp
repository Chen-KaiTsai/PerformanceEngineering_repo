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

    int k, h;
    h = 1;
    while (h < N)
        h = 3 * h + 1;
    h /= 3;

    #pragma omp parallel firstprivate(h) num_threads(NTHREAD)
    {
        while (h != 1)
        {
            #pragma omp for
            for (k = 0; k < h; k++)
            {
                int i, j, field_index, compare_result;
                char **v;

                for (i = k; i < N; i += h)
                {
                    v = input_data[i];
                    j = i;
                    bool flag;

                    while (true)
                    {
                        if (j - h < 0)
                            break;

                        for (field_index = 0; field_index < num_fields; field_index++)
                        {
                            compare_result = strcmp(input_data[j - h][field_index], v[field_index]);
                            if (compare_result > 0)
                            {
                                flag = true;
                                break;
                            }
                            else if (compare_result < 0)
                            {
                                flag = false;
                                break;
                            }
                        }

                        if (flag)
                        {
                            input_data[j] = input_data[j - h];
                            j -= h;
                            if (j <= h)
                                break;
                        }
                        else
                            break;
                    }
                    input_data[j] = v;
                }
            }
            h /= 2;
        }
    }

    //Insertion Sort at the final step

    char **v;
    int field_index, compare_result;

    for (i = 1; i < N; i++)
    {
        v = input_data[i];
        j = i;
        bool flag;

        while (true)
        {
            for (field_index = 0; field_index < num_fields; field_index++)
            {
                compare_result = strcmp(input_data[j - 1][field_index], v[field_index]);
                if (compare_result > 0)
                {
                    flag = true;
                    break;
                }
                else if (compare_result < 0)
                {
                    flag = false;
                    break;
                }
            }

            if (!flag)
                break;

            input_data[j] = input_data[j - 1];
            j--;
            if (j <= 0)
                break;
        }
        input_data[j] = v;
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