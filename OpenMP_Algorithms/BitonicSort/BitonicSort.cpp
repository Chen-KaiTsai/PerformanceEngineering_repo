#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <random>

#define N (8)

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

void bitonicMerge(int start_index, int end_index, int dir, int *input_data)
{
    if (dir == 1)
    {
        int num_elements = end_index - start_index + 1, temp;
        for (int j = num_elements / 2; j > 0; j = j / 2)
        {
            for (int i = start_index; i + j <= end_index; i++)
            {
                if (input_data[i] > input_data[i + j])
                {
                    temp = input_data[i + j];
                    input_data[i + j] = input_data[i];
                    input_data[i] = temp;
                }
            }
        }
    }
    else
    {
        int num_elements = end_index - start_index + 1, temp;
        for (int j = num_elements / 2; j > 0; j = j / 2)
        {
            for (int i = start_index; i + j <= end_index; i++)
            {
                if (input_data[i + j] > input_data[i])
                {
                    temp = input_data[i + j];
                    input_data[i + j] = input_data[i];
                    input_data[i] = temp;
                }
            }
        }
    }
}

int main()
{
    omp_set_dynamic(0);

    int NTHREAD;
    int uniform_loop;
    //int N;
    //scanf("%d", &NTHREAD);
    int testloop;
    for (testloop = 1; testloop < 10; testloop++)
    {

        //N = 1024 * 1024 * 64 * testloop;
        NTHREAD = testloop;

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

            for (int j = 2; j <= N; j *= 2)
            {
                #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
                for (int i = 0; i < N; i += j)
                {
                    if (((i / j) % 2) == 0)
                        bitonicMerge(i, i + j - 1, 1, input_data);
                    else
                        bitonicMerge(i, i + j - 1, 0, input_data);
                }
            }

            time_y = omp_get_wtime();

            //verify(input_data, N);

            
    for (i = 0; i < N; i++)
        printf("%d ", *(input_data + i));
    printf("\n");
    
            free(input_data);

            printf("DataSize = %d Num_Core = %d execute_time =  %lf\n", N, NTHREAD, time_y - time_x);
        }
    }
}