#include "main.h"

const uint32_t ZR = 0;
const uint32_t MV = 0x80000000;

alignas(32) const uint32_t c_Mask0[8]{ ZR, ZR, ZR, ZR, ZR, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask1[8]{ MV, ZR, ZR, ZR, ZR, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask2[8]{ MV, MV, ZR, ZR, ZR, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask3[8]{ MV, MV, MV, ZR, ZR, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask4[8]{ MV, MV, MV, MV, ZR, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask5[8]{ MV, MV, MV, MV, MV, ZR, ZR, ZR };
alignas(32) const uint32_t c_Mask6[8]{ MV, MV, MV, MV, MV, MV, ZR, ZR };
alignas(32) const uint32_t c_Mask7[8]{ MV, MV, MV, MV, MV, MV, MV, ZR };

const uint32_t* c_MaskMovLUT[8]
{
	c_Mask0, c_Mask1, c_Mask2, c_Mask3, c_Mask4, c_Mask5, c_Mask6, c_Mask7
};

/*
 * @brief Naive Matrix Multiplication with Arbitrary Size
*/
void matrixMulF32(float* C, const float* A, const float* B, size_t H, size_t W, size_t inner)
{
	for (size_t i = 0; i < H; ++i)
		for (size_t j = 0; j < W; ++j) {
			float sum = 0;
			for (size_t k = 0; k < inner; ++k) {
				float a = A[i * inner + k];
				float b = B[k * W + j];
				sum += a * b;
			}
			C[i * W + j] = sum;
		}
}

/*
 * @brief AVX2 version of Matrix Multiplication with Arbitrary Size
*/
void matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S)
{
	constexpr size_t numElementSIMD = 256 / 32;

	size_t numResidualCol = M % numElementSIMD;
	__m256i resMask = _mm256_load_si256((__m256i*)c_MaskMovLUT[numResidualCol]);

	#pragma omp parallel for num_threads(NTHREADS) schedule(static)
	for (size_t i = 0; i < N; ++i)
	{
		size_t j = 0;

		while (j + numElementSIMD <= M)
		{
			__m256 pC = _mm256_setzero_ps();

			for (size_t k = 0; k < S; ++k)
			{
				__m256 pA = _mm256_broadcast_ss(&A[i * S + k]);
				__m256 pB = _mm256_loadu_ps(&B[k * M + j]);

				pC = _mm256_fmadd_ps(pA, pB, pC);
			}

			_mm256_storeu_ps(&C[i * M + j], pC);
			j += numElementSIMD;
		}

		if (numResidualCol != 0)
		{
			__m256 pC = _mm256_setzero_ps();

			for (size_t k = 0; k < S; ++k)
			{
				__m256 pA = _mm256_broadcast_ss(&A[i * S + k]);
				__m256 pB = _mm256_maskload_ps(&B[k * M + j], resMask);

				pC = _mm256_fmadd_ps(pA, pB, pC);
			}

			_mm256_maskstore_ps(&C[i * M + j], resMask, pC);
		}
	}
}

/*
 * @brief VCL2 version of Matrix Multiplication with Arbitrary Size
*/
void matrixMulF32_VCL2(float* C, const float* A, const float* B, const int N, const int M, const int S)
{
	constexpr size_t numElementSIMD = 256 / 32;

	size_t numResidualCol = M % numElementSIMD;

	#pragma omp parallel for num_threads(NTHREADS) schedule(static)
	for (size_t i = 0; i < N; ++i)
	{
		size_t j = 0;

		while (j + numElementSIMD <= M)
		{
			Vec8f pC(0.0f);

			for (size_t k = 0; k < S; ++k)
			{
				Vec8f pA(A[i * S + k]);
				pC += pA * Vec8f().load((B + (k * M + j)));
			}

			pC.store((C + (i * M + j)));
			j += numElementSIMD;
		}

		if (numResidualCol != 0)
		{
			Vec8f pC(0.0f);

			for (size_t k = 0; k < S; ++k)
			{
				Vec8f pA(A[i * S + k]);
				pC += pA * Vec8f().load_partial(numResidualCol, (B + (k * M + j)));
			}

			pC.store_partial(numResidualCol, (C + (i * M + j)));
		}
	}
}
