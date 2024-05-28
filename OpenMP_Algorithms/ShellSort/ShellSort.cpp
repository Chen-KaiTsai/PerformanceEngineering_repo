#include <stdio.h> //All program are wrote in pure C except random generator part is in C++
#include <stdlib.h>
#include <omp.h>

#include <random> // C++ library

#define N (1024 * 1024 * 512)

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
        //N = 1024 * 1024 * 64 * testloop;

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

            //ShellSort

            //Should find a time to dig into the different gap.
            //https://en.wikipedia.org/wiki/Shellsort#Gap_sequences

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
                        int i, j, v;

                        for (i = k; i < N; i += h)
                        {
                            v = input_data[i];
                            j = i;
                            while (true)
                            {
                                if (j - h < 0)
                                    break;
                                if (input_data[j - h] > v)
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

            int j, v;
            for (i = 1; i < N; i++)
            {
                v = input_data[i];
                j = i;
                while (input_data[j - 1] > v)
                {
                    input_data[j] = input_data[j - 1];
                    j--;
                    if (j <= 0)
                        break;
                }
                input_data[j] = v;
            }

            time_y = omp_get_wtime();

            
            for (i = 0; i < N; i++)
                printf("%d ", *(input_data + i));
            printf("\n");
            

            free(input_data);

            printf("DataSize = %d Num_Core = %d execute_time =  %lf\n", N, NTHREAD, time_y - time_x);
        }
    }
}