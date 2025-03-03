#include "main.h"

/*
 * @warning When store pSum to gSum, false sharing will occur. 
 * However, the number of CPU threads is not too big to cause too much performance hit.
*/
void pSum(int* __restrict__ data, int* gSum, const int threadIdx) {
    int pSum = 0;
    constexpr int pSize = N / nThreads;
    int start = threadIdx * pSize;
    int end = (threadIdx + 1) * pSize;
    
    if (threadIdx == (nThreads - 1)) // boundary checking
        end = N;

    for (int i = start; i < end; ++i) {
        pSum += data[i];
    }

    gSum[threadIdx] = pSum;
}

int PALGO_c11::c11ParallelSum(int* __restrict__ data, const size_t &size) {
    int gSum[nThreads];
    std::vector<thread> tHandles;
    tHandles.reserve(nThreads);

    // create threads
    for (int t = 0; t < nThreads; ++t) {
        tHandles.emplace_back(thread(pSum, data, gSum, t));
    }

    // join threads
    int sum = 0;
    for (int t = 0; t < nThreads; ++t) {
        tHandles[t].join();
        sum += gSum[t];
    }

    return sum;
}

/*
 * @warning When store pSum to gSum, false sharing will occur. 
 * However, the number of CPU threads is not too big to cause too much performance hit.
 * @note barrier version which will return the final sum in gSum[0]
*/
void pSum_barrier(int* __restrict__ data, int* gSum, const int threadIdx) {
    int pSum = 0;
    constexpr int pSize = N / nThreads;
    int start = threadIdx * pSize;
    int end = (threadIdx + 1) * pSize;
    
    if (threadIdx == (nThreads - 1)) // boundary checking
        end = N;

    for (int i = start; i < end; ++i) {
        pSum += data[i];
    }

    gSum[threadIdx] = pSum;

    PALGO_c11::bStep.arrive_and_wait();

    int step = 2;
    for (int i = 1; i < nThreads; i *= 2) {
        if ((threadIdx % step) == 0) {
            if (threadIdx + i < nThreads)
                gSum[threadIdx] += gSum[threadIdx + i];
        }
        step *= 2; // cannot put it in the for loop since it need to be before barrier.

        PALGO_c11::bStep.arrive_and_wait();
    }
}

/* 
 * @warning Compile with std=c++20 is required. 
*/
int PALGO_c11::c11ParallelSum_barrier(int* __restrict__ data, const size_t &size) {
    int gSum[nThreads];
    std::vector<thread> tHandles;
    tHandles.reserve(nThreads);

    // create threads
    for (int t = 0; t < nThreads; ++t) {
        tHandles.emplace_back(thread(pSum_barrier, data, gSum, t));
    }

    return gSum[0];
}

void pSearch(int* __restrict__ data, const int &value, int* index, const int threadIdx) {
    *(index) = -1;
    constexpr int pSize = N / nThreads;
    int start = threadIdx * pSize;
    int end = (threadIdx + 1) * pSize;
    
    if (threadIdx == (nThreads - 1)) // boundary checking
        end = N;

    int count = 0;
    for (int i = start; i < end; ++i) {
        if ((count + 1) % 8 == 0) {
            if (PALGO_c11::flag)
                break;
        }

        if (data[i] == value) {
            *(index) = i;
            PALGO_c11::flag = true;
        }
        
        ++count;
    }
}

int PALGO_c11::c11ParallelLinearSearch(int* __restrict__ data, const size_t &size, const int &value) {
    PALGO_c11::flag = false;
    int index = -1;

    std::vector<thread> tHandles;
    tHandles.reserve(nThreads);

    // create threads
    for (int t = 0; t < nThreads; ++t) {
        tHandles.emplace_back(thread(pSearch, data, std::ref(value), &index, t));
    }

    return index;
}