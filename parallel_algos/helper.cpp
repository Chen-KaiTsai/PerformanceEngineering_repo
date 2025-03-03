#include "main.h"

void CPP_helper::initialize2Rand(int* __restrict__ data, const size_t &size, const int &max, const int &min) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution {min, max};
    for (int i = 0; i < size; ++i) {
        *(data + i) = distribution(generator);
    }
}

void CPP_helper::initialize2Dummy(int* __restrict__ data, const size_t &size) {
    for (int i = 0; i < size; ++i) {
        *(data + i) = 1;
    }
}
