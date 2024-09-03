#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#include <immintrin.h> // AVX, AVX2, FMA
#include "vectorclass.h"

#include "malloc2D.h"

#define MAX_VECTOR_SIZE 256

// Configuration
constexpr unsigned int NTHREADS = 128;

void matrixMulF32(float* C, const float* A, const float* B, size_t hA, size_t wA, size_t wB);
void matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S);
void matrixMulF32_VCL2(float* C, const float* A, const float* B, const int N, const int M, const int S);
