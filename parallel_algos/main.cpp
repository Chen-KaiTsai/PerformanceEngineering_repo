#include "main.h"

int main(int argc, char** argv)
{
    int sum = 0;
    std::unique_ptr<int[]> data = std::make_unique<int[]>(N);
    CPP_helper::initialize2Dummy(data.get(), N);

    sum = PALGO_c11::c11ParallelSum(data.get(), N);

    printf("Sum = %d\n\n", sum);

    return EXIT_SUCCESS;
}
