#include <iostream>
#include <algorithm>
#include <execution>
#include <memory>
#include <chrono>

#include <thread>

constexpr size_t DATASIZE = 1024 * 1024 * 1024;

int main(int argv, char** argc)
{
    size_t threadNum = std::thread::hardware_concurrency();
    printf("Hardware Thread Number : %zu\n\n", threadNum);

    std::unique_ptr<int[]> data = std::make_unique<int[]>(DATASIZE);

    for (int i = 0; i < DATASIZE; ++i) {
        data[i] = rand() % 100;
    }

    printf("\n\n------------------------------ Single Thread ------------------------------\n\n");
    auto start = std::chrono::steady_clock::now();
    std::sort(data.get(), data.get() + DATASIZE);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);

    for (int i = 0; i < DATASIZE; ++i) {
        data[i] = rand() % 100;
    }

    printf("\n\n------------------------------ Multi Thread / SIMD ------------------------------\n\n");
    start = std::chrono::steady_clock::now();
    std::sort(std::execution::par_unseq, data.get(), data.get() + DATASIZE);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);

    return 0;
}

