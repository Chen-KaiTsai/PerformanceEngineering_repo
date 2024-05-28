#include <iostream>
#include <omp.h>

using namespace std;

const int NUM_THREADS = 4;
const int N = 32;

int main(void)
{
    int x[N];
    for (int i = 0; i < N; ++i)
        x[i] = i + 1;

    int *suma;
    #pragma omp parallel num_threads(NUM_THREADS) //Keep threads alive for two stages
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            suma = new int[nthreads + 1];
            suma[0] = 0;
        }
        int sum = 0; //private sum
        #pragma omp for schedule(static, N / nthreads)
        for (int i = 0; i < N; ++i)
        {
            sum += x[i];
            x[i] = sum; //prefix scan
        }
        suma[ithread + 1] = sum; //partial_sum for each thread
        #pragma omp barrier

        for (int i = ithread * (N / nthreads); i < (ithread + 1) * (N / nthreads); ++i)
            for(int j = 1; j < ithread + 1; ++j)
                x[i] += suma[j];
    }
    delete[] suma;

    for (auto a : x)
        cout << a << "\t";
    cout << endl;
}