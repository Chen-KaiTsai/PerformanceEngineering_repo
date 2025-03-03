#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <chrono>

#include <thread>
#include <atomic>
#include <barrier>

#include <cstdio>
#include <cassert>

#include <omp.h>

using std::thread;

constexpr size_t N = 1024 * 1024 * 1024;
constexpr unsigned int nThreads = 32;

namespace CPP_helper 
{
    void initialize2Rand(int* __restrict__ data, const size_t &size, const int &max, const int &min);
    void initialize2Dummy(int* __restrict__ data, const size_t &size);
}

namespace PALGO_omp
{
    int ompParallelSum(int* __restrict__ data, const size_t &size);
}

namespace PALGO_c11
{
    std::barrier bStep {nThreads};
    int c11ParallelSum(int* __restrict__ data, const size_t &size);
    int c11ParallelSum_barrier(int* __restrict__ data, const size_t &size);

    std::atomic<bool> flag;
    int c11ParallelLinearSearch(int* __restrict__ data, const size_t &size, const int &value);
}
