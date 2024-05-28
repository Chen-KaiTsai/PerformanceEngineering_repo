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

void squareMatrixTranspose(int* X, int N)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < i; j++)
			std::swap(X[j * N + i], X[i * N + j]);
}

/*
* @brief Assume Row Major
* Reference : https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
*
*/
void matrixTranspose(float* src, float* dst, const int N, const int M)
{
	// TODO: can be parallized
#pragma omp parallel for
	for (int n = 0; n < N * M; n++) {
		int i = n / N; // # Row
		int j = n % N; // # Column

		dst[n] = src[M * j + i]; // TODO: Not contiguous
	}
}

/*
* @brief Squared Blocking version
*
* @param A src
* @param B dst
* @param lda matrix row size round up to blockSize
* @param ldb matrix column size round up to blockSize
* #param blockSize blockSize choosed
*
*/
void squareMatrixTransposeScalarBlock(float* A, float* B, const int ld, const int blockSize)
{
//#pragma omp parallel for
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			B[j * ld + i] = A[i * ld + j];
}


void squareMatrixTransposeBlocks(float* A, float* B, const int N, const int ld, const int blockSize)
{
//#pragma omp parallel for
	for (int i = 0; i < N; i += blockSize) {
		for (int j = 0; j < N; j += blockSize) {
			// send in block start point to function
			squareMatrixTransposeScalarBlock(&A[i * ld + j], &B[j * ld + i], ld, blockSize);
		}
	}
}

void matrixTransposeScalar4x4Block_Isse(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst)
{
	__m128 R0 = _mm_loadu_ps((src + ((blockIdxY + 0) * M + blockIdxX)));
	__m128 R1 = _mm_loadu_ps((src + ((blockIdxY + 1) * M + blockIdxX)));
	__m128 R2 = _mm_loadu_ps((src + ((blockIdxY + 2) * M + blockIdxX)));
	__m128 R3 = _mm_loadu_ps((src + ((blockIdxY + 3) * M + blockIdxX)));

	_MM_TRANSPOSE4_PS(R0, R1, R2, R3);

	_mm_storeu_ps((dst + ((blockIdxX + 0) * N + blockIdxY)), R0);
	_mm_storeu_ps((dst + ((blockIdxX + 1) * N + blockIdxY)), R1);
	_mm_storeu_ps((dst + ((blockIdxX + 2) * N + blockIdxY)), R2);
	_mm_storeu_ps((dst + ((blockIdxX + 3) * N + blockIdxY)), R3);
}
/*
* @brief Implementation Reference : https://stackoverflow.com/questions/16941098/fast-memory-transpose-with-sse-avx-and-openmp
* 
*/ 
void matrixTransposeScalar8x8Block_Iavx2(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst)
{
	__m256 R0 = _mm256_loadu_ps((src + ((blockIdxY + 0) * M + blockIdxX)));
	__m256 R1 = _mm256_loadu_ps((src + ((blockIdxY + 1) * M + blockIdxX)));
	__m256 R2 = _mm256_loadu_ps((src + ((blockIdxY + 2) * M + blockIdxX)));
	__m256 R3 = _mm256_loadu_ps((src + ((blockIdxY + 3) * M + blockIdxX)));
	__m256 R4 = _mm256_loadu_ps((src + ((blockIdxY + 4) * M + blockIdxX)));
	__m256 R5 = _mm256_loadu_ps((src + ((blockIdxY + 5) * M + blockIdxX)));
	__m256 R6 = _mm256_loadu_ps((src + ((blockIdxY + 6) * M + blockIdxX)));
	__m256 R7 = _mm256_loadu_ps((src + ((blockIdxY + 7) * M + blockIdxX)));

	__m256 T0 = _mm256_unpacklo_ps(R0, R1);
	__m256 T1 = _mm256_unpackhi_ps(R0, R1);
	__m256 T2 = _mm256_unpacklo_ps(R2, R3);
	__m256 T3 = _mm256_unpackhi_ps(R2, R3);
	__m256 T4 = _mm256_unpacklo_ps(R4, R5);
	__m256 T5 = _mm256_unpackhi_ps(R4, R5);
	__m256 T6 = _mm256_unpacklo_ps(R6, R7);
	__m256 T7 = _mm256_unpackhi_ps(R6, R7);

	__m256 TT0 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 TT1 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
	__m256 TT2 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 TT3 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));
	__m256 TT4 = _mm256_shuffle_ps(T4, T6, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 TT5 = _mm256_shuffle_ps(T4, T6, _MM_SHUFFLE(3, 2, 3, 2));
	__m256 TT6 = _mm256_shuffle_ps(T5, T7, _MM_SHUFFLE(1, 0, 1, 0));
	__m256 TT7 = _mm256_shuffle_ps(T5, T7, _MM_SHUFFLE(3, 2, 3, 2));

	R0 = _mm256_permute2f128_ps(TT0, TT4, 0x20);
	R1 = _mm256_permute2f128_ps(TT1, TT5, 0x20);
	R2 = _mm256_permute2f128_ps(TT2, TT6, 0x20);
	R3 = _mm256_permute2f128_ps(TT3, TT7, 0x20);
	R4 = _mm256_permute2f128_ps(TT0, TT4, 0x31);
	R5 = _mm256_permute2f128_ps(TT1, TT5, 0x31);
	R6 = _mm256_permute2f128_ps(TT2, TT6, 0x31);
	R7 = _mm256_permute2f128_ps(TT3, TT7, 0x31);

	_mm256_storeu_ps((dst + ((blockIdxX + 0) * N + blockIdxY)), R0);
	_mm256_storeu_ps((dst + ((blockIdxX + 1) * N + blockIdxY)), R1);
	_mm256_storeu_ps((dst + ((blockIdxX + 2) * N + blockIdxY)), R2);
	_mm256_storeu_ps((dst + ((blockIdxX + 3) * N + blockIdxY)), R3);
	_mm256_storeu_ps((dst + ((blockIdxX + 4) * N + blockIdxY)), R4);
	_mm256_storeu_ps((dst + ((blockIdxX + 5) * N + blockIdxY)), R5);
	_mm256_storeu_ps((dst + ((blockIdxX + 6) * N + blockIdxY)), R6);
	_mm256_storeu_ps((dst + ((blockIdxX + 7) * N + blockIdxY)), R7);
}

void matrixTransposeBorderBlock(int blockIdxY, int blockIdxX, const int N, const int M, const int blockSize, float* src, float* dst)
{
	int borderY = (blockIdxY + blockSize) < N ? (blockIdxY + blockSize) : N;
	int borderX = (blockIdxX + blockSize) < M ? (blockIdxX + blockSize) : M;
	for (int i = blockIdxY; i < borderY; i++) {
		for (int j = blockIdxX; j < borderX; j++) {
			dst[i + j * N] = src[i * M + j];
		}
	}
}

void matrixTransposeBlocks(float* src, float* dst, const int N, const int M, const int blockSize)
{
	for (int blockIdxY = 0; blockIdxY < N; blockIdxY += blockSize) {
		for (int blockIdxX = 0; blockIdxX < M; blockIdxX += blockSize) {
			if (blockIdxY + blockSize <= N && blockIdxX + blockSize <= M) {
				printf("Run a AVX2 Block (%3d, %3d)\n", blockIdxY, blockIdxX);
				matrixTransposeScalar8x8Block_Iavx2(blockIdxY, blockIdxX, N, M, blockSize, src, dst);
			}
			else {
				printf("Run a Border Block (%3d, %3d)\n", blockIdxY, blockIdxX);
				matrixTransposeBorderBlock(blockIdxY, blockIdxX, N, M, blockSize, src, dst);
			}
		}
	}
}

void matrixMulF32_square(float** a, float** b, float** c, size_t size)
{
	for (int i = 0; i < size; ++i)
		for (int k = 0; k < size; ++k)
			for (int j = 0; j < size; ++j)
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
}

/*
* @brief Sequantial Matrix Multiplication with Arbitrary Size
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

void matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S)
{
	constexpr size_t numElementSIMD = 256 / 32;

	size_t numResidualCol = M % numElementSIMD;
	__m256i resMask = _mm256_load_si256((__m256i*)c_MaskMovLUT[numResidualCol]);

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

		if (numResidualCol)
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
