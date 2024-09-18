#include <iostream>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

constexpr size_t DATASIZE = 1024 * 1024 * 1024;

int main(int argv, char** argc)
{
    thrust::default_random_engine eng(1024);
    thrust::uniform_int_distribution<int> dist;
    thrust::host_vector<float> hVec(DATASIZE);
    thrust::generate(hVec.begin(), hVec.end(), [&] {return dist(eng); });

    printf("\n\n------------------------------ Transfer data to GPU ------------------------------\n\n");
    
    auto start = std::chrono::steady_clock::now();
    thrust::device_vector<float> dVec = hVec;
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    
    printf("\n\n------------------------------ GPU Kernel ------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    thrust::sort(dVec.begin(), dVec.end());
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);

    printf("\n\n------------------------------ Transfer data from GPU ------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    thrust::copy(dVec.begin(), dVec.end(), hVec.begin());
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);

    return 0;
}

