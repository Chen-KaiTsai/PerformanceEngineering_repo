#include "main.h"

void convolution2D(int inputH, int inputW, int inChannel, int outputH, int outputW, int outChannel, int batchSize, int stride, int kSize, int padSize, float* X, float* W,float* B, float* Y)
{
	int wSubSize = kSize * kSize * inChannel;
	int xOneBatchSize = inputH * inputW * inChannel;
	int xMapSize = inputH * inputW;
	int yOneBatchSize = outputH * outputW * outChannel;
	int yMapSize = outputH * outputW;

	float sum;
	int indexW;

	for (int N = 0; N < batchSize; ++N)
	{
		for (int h = 0; h < outputH; ++h)
		{
			for (int w = 0; w < outputW; ++w)
			{
				for (int cOut = 0; cOut < outChannel; ++cOut)
				{
					sum = 0.0f;
					indexW = 0;
					for (int cIn = 0; cIn < inChannel; ++cIn)
					{
						for (int kh = 0; kh < kSize; ++kh)
						{
							for (int kw = 0; kw < kSize; ++kw, ++indexW)
							{
								int hp = h * stride + kh - padSize;
								int wp = w * stride + kw - padSize;
								if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW)
									sum += W[cOut * wSubSize + indexW] * X[N * xOneBatchSize + cIn * xMapSize + hp * inputW + wp];
							}
						}
					}
					if (B != nullptr) {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum + B[cOut];
					}
					else {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum;
					}
				}
			}
		}
	}
}

/*
* @brief Only Support stride = 1 and odd size kernel [TODO require test and review]
*/
void convolution2D_Iavx(int inputH, int inputW, int inChannel, int outputH, int outputW, int outChannel, int batchSize, int stride, int kSize, int padSize, float* X, float* W, float* B, float* Y)
{
	int wSubSize = kSize * kSize * inChannel;
	int xOneBatchSize = inputH * inputW * inChannel;
	int xMapSize = inputH * inputW;
	int yOneBatchSize = outputH * outputW * outChannel;
	int yMapSize = outputH * outputW;

	constexpr size_t numElemntSIMD = 256 / 32;

	for (int N = 0; N < batchSize; ++N)
	{
		for (int h = 0; h < outputH; ++h)
		{
			int w = 0;
			while (w + numElemntSIMD <= outputW - (kSize / 2))
			{
				// inner case [use SIMD]
				if ((h * stride + 0 - padSize) >= 0 && (h * stride + (kSize - 1) - padSize) < inputH && (w * stride + 0 - padSize) >= 0) {
					for (int cOut = 0; cOut < outChannel; ++cOut)
					{
						__m256 pSum = _mm256_setzero_ps();
						int indexW = 0;
						for (int cIn = 0; cIn < inChannel; ++cIn)
						{
							for (int kh = 0; kh < kSize; ++kh)
							{
								for (int kw = 0; kw < kSize; ++kw, ++indexW)
								{
									int hp = h * stride + kh - padSize;
									int wp = w * stride + kw - padSize;
									__m256 pW = _mm256_broadcast_ss(&W[cOut * wSubSize + indexW]);
									__m256 pX = _mm256_loadu_ps(&X[N * xOneBatchSize + cIn * xMapSize + hp * inputW + wp]);
									pSum = _mm256_fmadd_ps(pW, pX, pSum);
								}
							}
						}
						if (B != nullptr) {
							__m256 pB = _mm256_broadcast_ss(&B[cOut]);
							pSum = _mm256_add_ps(pSum, pB);
							_mm256_store_ps(&Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w], pSum);
						}
						else {
							_mm256_store_ps(&Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w], pSum);
						}
					}
				}
				// border case [TODO : use mask to apply SIMD]
				else
				{
					for (int e = 0; e < numElemntSIMD; ++e) // sequentially count numElementSIMD elements
					{
						for (int cOut = 0; cOut < outChannel; ++cOut)
						{
							float sum = 0.0f;
							int indexW = 0;
							for (int cIn = 0; cIn < inChannel; ++cIn)
							{
								for (int kh = 0; kh < kSize; ++kh)
								{
									for (int kw = 0; kw < kSize; ++kw, ++indexW)
									{
										int hp = h * stride + kh - padSize;
										int wp = (w + e) * stride + kw - padSize;
										if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW)
											sum += W[cOut * wSubSize + indexW] * X[N * xOneBatchSize + cIn * xMapSize + hp * inputW + wp];
									}
								}
							}
							if (B != nullptr) {
								Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + (w + e)] = sum + B[cOut];
							}
							else {
								Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + (w + e)] = sum;
							}
						}
					}
				}
			}
			// Residual Sequentially [TODO : can be further improve with SSE]
			for (; w < outputW; ++w)
			{
				for (int cOut = 0; cOut < outChannel; ++cOut)
				{
					float sum = 0.0f;
					int indexW = 0;
					for (int cIn = 0; cIn < inChannel; ++cIn)
					{
						for (int kh = 0; kh < kSize; ++kh)
						{
							for (int kw = 0; kw < kSize; ++kw, ++indexW)
							{
								int hp = h * stride + kh - padSize;
								int wp = w * stride + kw - padSize;
								if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW)
									sum += W[cOut * wSubSize + indexW] * X[N * xOneBatchSize + cIn * xMapSize + hp * inputW + wp];
							}
						}
					}
					if (B != nullptr) {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum + B[cOut];
					}
					else {
						Y[N * yOneBatchSize + cOut * yMapSize + h * outputW + w] = sum;
					}
				}
			}
		}
	}
}
