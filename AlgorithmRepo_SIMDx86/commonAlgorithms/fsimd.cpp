#include "main.h"

bool fsimd::calcMinMaxU8_Iavx(uint8_t* xMin, uint8_t* xMax, const uint8_t* X, size_t size)
{
	if (size == 0 || ((size % 16) != 0)) {
		perror("Data size error");
		return false;
	}

	if (!AlignedMem::IsAligned(X, 16)) {
		perror("Data not aligned");
		return false;
	}

	constexpr size_t numElementSIMD = 128 / 8;

	__m128i pMin = _mm_set1_epi8((char)0xff);
	__m128i pMax = _mm_setzero_si128();


	// [TODO] : OpenMP Version => Require local pMin, pMax and reduction on thread local pMin, pMax
	for (size_t i = 0; i < size / numElementSIMD; i++)
	{
		__m128i pX = _mm_load_si128((__m128i*) & X[i * numElementSIMD]);
		
		pMin = _mm_min_epu8(pX, pMin);
		pMax = _mm_max_epu8(pX, pMax);
	}


	// Reduction on a packed data register
	__m128i temp1, temp2, temp3, temp4;
	__m128i pR1, pR2, pR3, pR4;

	temp1 = _mm_srli_si128(pMin, 8);
	pR1 = _mm_min_epu8(pMin, temp1);
	temp2 = _mm_srli_si128(pR1, 4);
	pR2 = _mm_min_epu8(pR1, temp2);
	temp3 = _mm_srli_si128(pR2, 2);
	pR3 = _mm_min_epu8(pR2, temp3);
	temp4 = _mm_srli_si128(pR3, 1);
	pR4 = _mm_min_epu8(pR3, temp4);

	*xMin = (uint8_t)_mm_extract_epi8(pR4, 0);

	temp1 = _mm_srli_si128(pMax, 8);
	pR1 = _mm_max_epu8(pMax, temp1);
	temp2 = _mm_srli_si128(pR1, 4);
	pR2 = _mm_max_epu8(pR1, temp2);
	temp3 = _mm_srli_si128(pR2, 2);
	pR3 = _mm_max_epu8(pR2, temp3);
	temp4 = _mm_srli_si128(pR3, 1);
	pR4 = _mm_max_epu8(pR3, temp4);

	*xMax = (uint8_t)_mm_extract_epi8(pR4, 0);

	return true;
}

bool fsimd::calcMeanU8_Iavx(double* xMean, uint64_t* xSum, const uint8_t* X, size_t size)
{
	constexpr size_t numElementMax = 1024 * 1024 * 64;

	if (size == 0 || size > numElementMax) {
		perror("Data size error");
		return false;
	}
	if ((size % 64) != 0) {
		perror("Data size error");
		return false;
	}
	if (!AlignedMem::IsAligned(X, 16)) {
		perror("Data not aligned");
		return false;
	}

	constexpr size_t numElementSIMD = 128 / 8;

	__m128i pZero = _mm_setzero_si128();
	__m128i pPixelSums_u32 = _mm_setzero_si128();

	for (size_t i = 0; i < size; i += numElementSIMD * 4)
	{
		__m128i pPixel_u8, pPixel_lo_u16, pPixel_hi_u16;
		__m128i pPixelSums_u16 = _mm_setzero_si128();

		// process 16 elements a time and promote 32 bit should make sure that 4 times of unrolling will not overflow the 16bits value. 

		// x[i : i+15]
		pPixel_u8 = _mm_load_si128((__m128i*) & X[i]);
		pPixel_lo_u16 = _mm_unpacklo_epi8(pPixel_u8, pZero);
		pPixel_hi_u16 = _mm_unpackhi_epi8(pPixel_u8, pZero);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_lo_u16);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_hi_u16);

		// x[i+16 : i+31]
		pPixel_u8 = _mm_load_si128((__m128i*) & X[i + 16]);
		pPixel_lo_u16 = _mm_unpacklo_epi8(pPixel_u8, pZero);
		pPixel_hi_u16 = _mm_unpackhi_epi8(pPixel_u8, pZero);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_lo_u16);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_hi_u16);
		
		// x[i+32 : i+47]
		pPixel_u8 = _mm_load_si128((__m128i*) & X[i + 32]);
		pPixel_lo_u16 = _mm_unpacklo_epi8(pPixel_u8, pZero);
		pPixel_hi_u16 = _mm_unpackhi_epi8(pPixel_u8, pZero);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_lo_u16);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_hi_u16);

		// x[i+48 : i+63]
		pPixel_u8 = _mm_load_si128((__m128i*) & X[i + 48]);
		pPixel_lo_u16 = _mm_unpacklo_epi8(pPixel_u8, pZero);
		pPixel_hi_u16 = _mm_unpackhi_epi8(pPixel_u8, pZero);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_lo_u16);
		pPixelSums_u16 = _mm_add_epi16(pPixelSums_u16, pPixel_hi_u16);

		// promote
		__m128i pPixelSum_lo_u32 = _mm_unpacklo_epi16(pPixelSums_u16, pZero);
		__m128i pPixelSum_hi_u32 = _mm_unpackhi_epi16(pPixelSums_u16, pZero);
		pPixelSums_u32 = _mm_add_epi32(pPixelSums_u32, pPixelSum_lo_u32);
		pPixelSums_u32 = _mm_add_epi32(pPixelSums_u32, pPixelSum_hi_u32);
	}

	uint64_t pixelSum = _mm_extract_epi32(pPixelSums_u32, 0);
	pixelSum += _mm_extract_epi32(pPixelSums_u32, 1);
	pixelSum += _mm_extract_epi32(pPixelSums_u32, 2);
	pixelSum += _mm_extract_epi32(pPixelSums_u32, 3);

	*xSum = pixelSum;
	*xMean = static_cast<double>(pixelSum) / static_cast<double>(size);

	return true;
}

bool fsimd::convertRGBtoGrayScale_Iavx2(uint8_t* pImageGS, const RGB32* pImageRGB, size_t size, const float coef[4])
{
	constexpr size_t numPixelsMax = 1024 * 1024 * 256;

	if (size > numPixelsMax) {
		perror("Image size error");
		return false;
	}
	if (size % 8 != 0) {
		perror("Image size error");
		return false;
	}
	if (!AlignedMem::IsAligned(pImageGS, 32)) {
		perror("Gray Scale Image not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(pImageRGB, 32)) {
		perror("RGB Image not aligned");
		return false;
	}
	if (coef[0] < 0.0f || coef[1] < 0.0f || coef[2] < 0.0f) {
		perror("Coefficient error");
		return false;
	}
	
	__m256 p0p5_f32 = _mm256_set1_ps(0.5f);
	__m256 p255p0_f32 = _mm256_set1_ps(255.0f);
	__m256i pZero = _mm256_setzero_si256();
	// AND this mask will set zero to all bytes except the right most byte
	__m256i p0Mask_u32 = _mm256_set1_epi32(0x000000ff);
	
	__m256 pCoefR = _mm256_set1_ps(coef[0]);
	__m256 pCoefG = _mm256_set1_ps(coef[1]);
	__m256 pCoefB = _mm256_set1_ps(coef[2]);

	constexpr size_t numElementSIMD = 256 / (8 * 4);

	for (size_t i = 0; i < size; i += numElementSIMD)
	{
		// load 8 pixels each with 4 channels
		__m256i pPixels = _mm256_load_si256((__m256i*) & pImageRGB[i]);

		// extract color channels and size promote to U32 (promote by zero out top bits [all pixels > 0])
		__m256i pPixelsR = pPixels;
		__m256i pPixelsG = _mm256_srli_epi32(pPixels, 8);
		__m256i pPixelsB = _mm256_srli_epi32(pPixels, 16);

		__m256i pPixelsR_u32 = _mm256_and_si256(pPixelsR, p0Mask_u32);
		__m256i pPixelsG_u32 = _mm256_and_si256(pPixelsG, p0Mask_u32);
		__m256i pPixelsB_u32 = _mm256_and_si256(pPixelsB, p0Mask_u32);

		// convert elements from U32 to F32
		__m256 pPixelsR_f32 = _mm256_cvtepi32_ps(pPixelsR_u32);
		__m256 pPixelsG_f32 = _mm256_cvtepi32_ps(pPixelsG_u32);
		__m256 pPixelsB_f32 = _mm256_cvtepi32_ps(pPixelsB_u32);

		pPixelsR_f32 = _mm256_mul_ps(pPixelsR_f32, pCoefR);
		pPixelsG_f32 = _mm256_mul_ps(pPixelsG_f32, pCoefG);
		pPixelsB_f32 = _mm256_mul_ps(pPixelsB_f32, pCoefB);

		__m256 temp1_f32 = _mm256_add_ps(pPixelsR_f32, pPixelsG_f32);
		__m256 pPixelsGS_f32 = _mm256_add_ps(temp1_f32, pPixelsB_f32);

		// clip grapscale pixel values [rounding]
		pPixelsGS_f32 = _mm256_add_ps(pPixelsGS_f32, p0p5_f32);
		pPixelsGS_f32 = _mm256_min_ps(pPixelsGS_f32, p255p0_f32);

		// convert grayscale values from F32 to U8
		__m256i pPixelsGS_u32 = _mm256_cvtps_epi32(pPixelsGS_f32);
		__m256i pPixelsGS_u16 = _mm256_packus_epi32(pPixelsGS_u32, pZero);
		__m256i temp2_u16 = _mm256_permute4x64_epi64(pPixelsGS_u16, 0b01011000);
		__m256i pPixelsGS_u8 = _mm256_packus_epi16(temp2_u16, pZero);

		uint64_t pixelsGS = _mm256_extract_epi64(pPixelsGS_u8, 0);
		*((uint64_t*)&pImageGS[i]) = pixelsGS;
	}

	return true;
}

inline double sumF64x4(__m256d X)
{
	double sum;
	__m128d temp0 = _mm256_extractf128_pd(X, 0);
	__m128d temp1 = _mm256_extractf128_pd(X, 1);
	__m128d temp2 = _mm_add_pd(temp0, temp1);
	__m128d temp3 = _mm_hadd_pd(temp2, temp2);

	_mm_store_sd(&sum, temp3);

	return sum;
}

bool fsimd::calcLeastSquare_Iavx2(double* m, double* b, const double* X, const double* Y, size_t size)
{
	if (size < 2) {
		perror("Data size error");
		return false;
	}
	if (!AlignedMem::IsAligned(X, 32)) {
		perror("Data X not aligned");
		return false;
	}
	if (!AlignedMem::IsAligned(Y, 32)) {
		perror("Data Y not aligned");
		return false;
	}

	*m = 0.0;
	*b = 0.0;

	__m256d pSumX = _mm256_setzero_pd();
	__m256d pSumY = _mm256_setzero_pd();
	__m256d pSumXX = _mm256_setzero_pd();
	__m256d pSumXY = _mm256_setzero_pd();

	constexpr size_t numElementSIMD = 256 / 64;
	constexpr double epsilon = 1.0e-12;

	size_t i = 0;

	// reduce in packs
	for (; size - i >= numElementSIMD; i += numElementSIMD)
	{
		__m256d pX = _mm256_load_pd(&X[i]);
		__m256d pY = _mm256_load_pd(&Y[i]);

		pSumX = _mm256_add_pd(pSumX, pX);
		pSumY = _mm256_add_pd(pSumY, pY);

		pSumXX = _mm256_fmadd_pd(pX, pX, pSumXX);
		pSumXY = _mm256_fmadd_pd(pX, pY, pSumXY);
	}

	double sumX = sumF64x4(pSumX);
	double sumY = sumF64x4(pSumY);
	double sumXX = sumF64x4(pSumXX);
	double sumXY = sumF64x4(pSumXY);

	// sequentially reduce left overs
	for (; i < size; ++i)
	{
		sumX += X[i];
		sumY += Y[i];
		sumXX += X[i] * X[i];
		sumXY += X[i] * Y[i];
	}

	double denom = size * sumXX - sumX * sumX;

	if (abs(denom) < epsilon) {
		denom = epsilon;
	}

	*m = (size * sumXY - sumX * sumY) / denom;
	*b = (sumXX * sumY - sumX * sumXY) / denom;

	return true;
}

bool fsimd::matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S)
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
