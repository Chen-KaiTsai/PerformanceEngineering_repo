#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <random>

#define N (1024 * 1024 * 512)

void merge(int *input_data, int i1, int j1, int i2, int j2)
{
    int i, j, k;
    i = i1;
    j = i2;
    k = 0;

    int temp_size = j2 - i1 + 1;
    int *temp = (int *)malloc(temp_size * sizeof(int));

    while (i <= j1 && j <= j2)
    {
        if (input_data[i] < input_data[j])
            temp[k++] = input_data[i++];
        else
            temp[k++] = input_data[j++];
    }

    while (i <= j1)
        temp[k++] = input_data[i++];

    while (j <= j2)
        temp[k++] = input_data[j++];

    for (i = i1, j = 0; i <= j2; i++, j++)
        input_data[i] = temp[j];

    free(temp);
}

void serial_mergesort(int *input_data, int i, int j)
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

void mergesort(int *input_data, int i, int j, int threads)
{
    int mid;
    if (threads == 1)
        serial_mergesort(input_data, i, j);
    else if (i < j)
    {
        //if (j - i <= N / threads) //need to be tuned.
        //{
        //    serial_mergesort(input_data, i, j);
        //    return;
        //}

        mid = (i + j) / 2;

        /*
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    //printf("Thread %d begins recursive call\n", omp_get_thread_num());
                    mergesort(input_data, i, mid);
                }
                
                #pragma omp task
                {
                    //printf("Thread %d begins recursive call\n", omp_get_thread_num());
                    mergesort(input_data, mid + 1, j);
                }
                
                #pragma omp taskwait
                {
                    //printf("Thread %d merge call\n", omp_get_thread_num());
                    merge(input_data, i, mid, mid + 1, j);
                }
            }
        }    
        */

        #pragma omp parallel sections
        {
//printf("Thread %d begins recursive section\n", omp_get_thread_num());
            #pragma omp section
            {
                //printf("Thread %d begins recursive call\n", omp_get_thread_num());
                mergesort(input_data, i, mid, threads / 2);
            }

            #pragma omp section
            {
                //printf("Thread %d begins recursive call\n", omp_get_thread_num());
                mergesort(input_data, mid + 1, j, threads / 2);
            }
        }

        merge(input_data, i, mid, mid + 1, j);
    }
}

void verify(int *A, int n)
{
    int failcount = 0;
    for (int iter = 0; iter < n - 1; iter++)
    {
        if (A[iter] > A[iter + 1])
        {
            failcount++;
        }
    }
    printf("\nFail count: %d\n", failcount);
}

int main()
{
    int NTHREAD;
    //int N;
    //scanf("%d", &NTHREAD);

    int testloop;
    int uniform_loop;
    for (testloop = 1; testloop < 10; testloop++)
    {
        NTHREAD = testloop;
        //N= 1024 * 1024 * 64 * testloop;
        omp_set_num_threads(NTHREAD);
        omp_set_dynamic(0);
        omp_set_nested(1);

        for (uniform_loop = 1024; uniform_loop > 1; uniform_loop /= 8)
        {
            int *input_data = (int *)malloc(N * sizeof(int));

            // C++ code the generator
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution{0, (N / uniform_loop) - 1};

            int i;
            for (i = 0; i < N; i++)
                *(input_data + i) = distribution(generator);

            printf("uniform_int_distribution with range 0 ~ %d\n", ((N / uniform_loop) - 1));
            //end with the generator
            //int *input_data = (int *)malloc(N * sizeof(int));
            //int i;
            //for (i = 0; i < N; i++)
            //    *(input_data + i) = rand();

            double time_x, time_y;

            time_x = omp_get_wtime();

            mergesort(input_data, 0, N - 1, NTHREAD);

            time_y = omp_get_wtime();

            free(input_data);

            printf("DataSize = %d Num_Core = %d execute_time =  %lf\n", N, NTHREAD, time_y - time_x);
        }
    }

    //verify(input_data, N);

    /*
    for (i = 0; i < N; i++)
        printf("%d ", *(input_data + i));
    printf("\n");
    */
}