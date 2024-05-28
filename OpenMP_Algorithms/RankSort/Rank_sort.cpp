#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <random>

#define N (1024)

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

int main(void)
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
            int *input_data = (int *)malloc(N * sizeof(int));

            // C++ code the generator
            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution{0, (N / uniform_loop) - 1};
            //std::uniform_int_distribution<int> distribution{0, 10};

            int i;
            for (i = 0; i < N; i++)
                *(input_data + i) = distribution(generator);

            printf("uniform_int_distribution with range 0 ~ %d with %d threads\n", ((N / uniform_loop) - 1), NTHREAD);
            //end with the generator

            int j, rank;

            int *sorted_data = (int *)malloc(N * sizeof(int));
            for (i = 0; i < N; i++)
                *(sorted_data + i) = -1;

            double time_x, time_y;

            time_x = omp_get_wtime();

            #pragma omp parallel for private(rank, j) num_threads(NTHREAD) schedule(guided)
            for (i = 0; i < N; i++)
            {
                rank = 0;
                for (j = 0; j < N; j++)
                    if (input_data[i] > input_data[j])
                        rank++;
                //sorted_data[rank] = input_data[i];
                for(;;)
                {
                    if(sorted_data[rank] == input_data[i])
                        rank++;
                    else
                    {
                        sorted_data[rank] = input_data[i];
                        break;
                    }
                        
                }
            }

            time_y = omp_get_wtime();

            
    //for (i = 0; i < N; i++)
    //    printf("%d ", sorted_data[i]);
    //printf("\n");
    
            verify(sorted_data, N);
            free(input_data);

            //for(int i = 0; i < N; i++)
            //    free(sorted_data[i]);

            free(sorted_data);

            printf("%lf\n", time_y - time_x);
        }
    }
}