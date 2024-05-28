#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <random>

#define N (1024 * 32 * 8)

struct value_index
{
    int val;
    int index;
};

#pragma omp declare reduction(maximum : struct value_index : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)

void verify(int *A, int n)
{
    int failcount = 0;
    for (int iter = 0; iter < n - 1; iter++)
    {
        if (A[iter] < A[iter + 1])
        {
            failcount++;
        }
    }
    printf("\nFail count: %d\n", failcount);
}

int main()
{
    int NTHREAD, testloop, uniform_loop;
    //int N;
    //scanf("%d", &NTHREAD);

    for (testloop = 1; testloop < 10; testloop++)
    {
        NTHREAD = testloop;

        for (uniform_loop = 16; uniform_loop > 1; uniform_loop /= 2)
        {
            //N = 1024 * 32 * testloop;

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

            //Selection Sort
            int startpos;
            for (startpos = 0; startpos < N; startpos++)
            {
                struct value_index max;
                max.val = input_data[startpos];
                max.index = startpos;

                #pragma omp parallel for reduction(maximum:max) num_threads(NTHREAD)
                for (i = startpos + 1; i < N; i++)
                {
                    if (input_data[i] > max.val)
                    {
                        max.val = input_data[i];
                        max.index = i;
                    }
                }

                int temp = input_data[startpos];
                input_data[startpos] = input_data[max.index];
                input_data[max.index] = temp;
            }

            time_y = omp_get_wtime();

            //verify(input_data, N);

            /*
    for (i = 0; i < N; i++)
        printf("%d ", *(input_data + i));
    printf("\n");
    */
            free(input_data);

            printf("%lf\n", time_y - time_x);
        }
    }
}