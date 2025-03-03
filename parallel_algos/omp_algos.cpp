#include "main.h"

int PALGO_omp::ompParallelSum(int* __restrict__ data, const size_t &size) {
    int sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(nThreads)
    for (int i = 0; i < size; i++) {
        sum = sum + data[i];
    }

    return sum;
}
