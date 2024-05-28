#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <random>

#define N (1024 * 32 * 8)

int main()
{
    int NTHREAD, testloop, uniform_loop;
    //int N;
    //scanf("%d", &NTHREAD);

    for (testloop = 1; testloop < 10; testloop++)
    {
        NTHREAD = testloop;
        //N = 1024 * 32 * testloop;

        for (uniform_loop = 16; uniform_loop > 1; uniform_loop /= 2)
        {

            /*
    int *input_data = (int *)malloc(N * sizeof(int));
    int i;
    for (i = 0; i < N; i++)
        *(input_data + i) = rand();
    */

            int *input_data = (int *)malloc(N * sizeof(int));

            // C++ code the generator
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution{0, (N / uniform_loop) - 1};

            int i;
            for (i = 0; i < N; i++)
                *(input_data + i) = distribution(generator);

            printf("uniform_int_distribution with range 0 ~ %d\n", ((N / uniform_loop) - 1));
            //end with the generator

            double time_x, time_y;

            time_x = omp_get_wtime();

            int exch = 1;
            while (exch)
            {
                exch = 0;
                #pragma omp parallel num_threads(NTHREAD)
                {
                    int temp;
                    #pragma omp for
                    for (i = 0; i < N - 1; i += 2)
                    {
                        if (input_data[i] > input_data[i + 1])
                        {
                            temp = input_data[i];
                            input_data[i] = input_data[i + 1];
                            input_data[i + 1] = temp;
                            exch = 1;
                        }
                    }

                    #pragma omp for
                    for (i = 1; i < N - 1; i += 2)
                    {
                        if (input_data[i] > input_data[i + 1])
                        {
                            temp = input_data[i];
                            input_data[i] = input_data[i + 1];
                            input_data[i + 1] = temp;
                            exch = 1;
                        }
                    }
                }
            }

            time_y = omp_get_wtime();

            free(input_data);

            printf("DataSize = %d Num_Core = %d execute_time =  %lf\n", N, NTHREAD, time_y - time_x);
        }
    }
    /*
    for (i = 0; i < N; i++)
        printf("%d ", *(input_data + i));
    printf("\n");
    */
}