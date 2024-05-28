#include "main.h"

bool fwsimd::ReLU_Iavx2(const float* X, float* Y, size_t size)
{
	if (!AlignedMem::IsAligned(X, 32)) {
		perror("Data X not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(Y, 32)) {
		perror("Data Y not aligned");
		return false;
	}

	__m256 pZero = _mm256_setzero_ps();

	size_t i = 0;
	constexpr size_t numElementSIMD = 256 / 32;

	for (; size - i >= numElementSIMD; i += numElementSIMD) 
	{
		__m256 pX = _mm256_load_ps(&X[i]);
		__m256 temp = _mm256_max_ps(pX, pZero);
		_mm256_store_ps(&Y[i], temp);
	}

	for (; i < size; ++i) {
		Y[i] = max(X[i], 0.0f);
	}

	return true;
}

bool fwsimd::LeakyReLU_Iavx2(const float* X, float* Y, const float negativeSlope, size_t size)
{
	if (!AlignedMem::IsAligned(X, 32)) {
		perror("Data X not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(Y, 32)) {
		perror("Data Y not aligned");
		return false;
	}

	__m256 pZero = _mm256_setzero_ps();
	__m256 pNegativeSlope = _mm256_set1_ps(negativeSlope);

	size_t i = 0;
	constexpr size_t numElementSIMD = 256 / 32;

	for (; size - i >= numElementSIMD; i += numElementSIMD)
	{
		__m256 pX = _mm256_load_ps(&X[i]);
		__m256 temp1 = _mm256_max_ps(pX, pZero);
		__m256 temp2 = _mm256_min_ps(pX, pZero);
		__m256 temp3 = _mm256_mul_ps(temp2, pNegativeSlope);
		__m256 temp4 = _mm256_add_ps(temp1, temp3);
		_mm256_store_ps(&Y[i], temp4);
	}

	for (; i < size; ++i) {
		Y[i] = max(X[i], 0.0f) + negativeSlope * min(X[i], 0.0f);
	}

	return true;
}

bool fwsimd::Add_Iavx2(const float* X1, const float* X2, float* Y, size_t size)
{
	if (!AlignedMem::IsAligned(X1, 32)) {
		perror("Data X not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(X2, 32)) {
		perror("Data X not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(Y, 32)) {
		perror("Data Y not aligned");
		return false;
	}

	size_t i = 0;
	constexpr size_t numElementSIMD = 256 / 32;

	for (; size - i >= numElementSIMD; i += numElementSIMD)
	{
		__m256 pX1 = _mm256_load_ps(&X1[i]);
		__m256 pX2 = _mm256_load_ps(&X2[i]);
		__m256 temp = _mm256_add_ps(pX1, pX2);
		_mm256_store_ps(&Y[i], temp);
	}

	for (; i < size; ++i) {
		Y[i] = X1[i] + X2[i];
	}

	return true;
}
