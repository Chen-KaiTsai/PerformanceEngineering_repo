#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <random>

#include <immintrin.h> //AVX, AVX2, FMA

#include "malloc2D.h"

void squareMatrixTranspose(int* X, int N);

void matrixTranspose(float* src, float* dst, const int N, const int M);

void squareMatrixTransposeScalarBlock(float* A, float* B, const int ld, const int blockSize);
void squareMatrixTransposeBlocks(float* A, float* B, const int N, const int ld, const int blockSize);

void matrixTransposeScalar4x4Block_Isse(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst);
void matrixTransposeScalar8x8Block_Iavx2(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst);
void matrixTransposeBorderBlock(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst);
void matrixTransposeBlocks(float* src, float* dst, const int N, const int M, const int blockSize);

void matrixMulF32_square(float** a, float** b, float** c, size_t size);
void matrixMulF32(float* C, const float* A, const float* B, size_t hA, size_t wA, size_t wB);
void matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S);

inline int roundUp(int X, int M) {
	return ((X + (M - 1)) / M) * M;
}
inline int roundUp_PW2(int X, int M) {
	return ((X + (M - 1)) & -M);
}
inline int roundUp_negative(int X, int M) {
	int isPositive = static_cast<int>(X >= 0);
	return ((X + (isPositive * (M - 1))) / M) * M;
}
