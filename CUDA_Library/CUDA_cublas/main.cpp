#include <iostream>
#include <iomanip> // iomanip
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cudnn.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float* getManagedMatrix(const int N, const int M)
{
    float* data = nullptr;
    cudaMallocManaged((void**)&data, N * M * sizeof(float));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            data[IDX2C(j, i, M)] = 1.0f;//(float)rand() / RAND_MAX;

    return data;
}

void printMatrix(const float* data, const int N, const int M)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << data[IDX2C(j, i, M)];
        }
        std::cout << std::endl;
    }
    std::cout << "\n\n";
}

int main(int argc, char** argv)
{
    cublasHandle_t handle;
    int N = 8, M = 4, S = 5;
    float alpha = 0.1f, beta = 0.0f;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed\n\n";
        exit(EXIT_FAILURE);
    }
    float* A = getManagedMatrix(N, S);
    float* B = getManagedMatrix(S, M);
    float* C = getManagedMatrix(N, M);
    cublasSgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, S,
        &alpha,
        A, N,
        B, S,
        &beta,
        C, N
    );
    cudaDeviceSynchronize();
    printMatrix(A, N, S);
    printMatrix(B, S, M);
    printMatrix(C, N, M);
    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

