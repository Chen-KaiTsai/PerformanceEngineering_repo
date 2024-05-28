/*
* This code repository contain some modified codes by a book.
* https://github.com/Apress/modern-parallel-programming-cpp-assembly
* Please refer to the book for detail.
* 
* Please compile with the `Include` folder in the code repository of the book.
* 
* The naming convention of functions will adapt from the book.
* However, naming convention of variables is redesigned
*	p prefix => packed
*	...
* 
* Please refer to the following HackMD for other detail
* [TODO] https://hackmd.io/6h4V3xR6TIaa5xx6dzML1Q#Convolution-1D2D-256-bits
* [TODO]
*/

#pragma once

// Standard Libraries
#include <iostream>
#include <cstdio>
#include <cstdlib>

// SIMD Intrinscs
#include <immintrin.h> //AVX, AVX2, FMA

// Helper functions provide by book [modern-parallel-programming-cpp-assembly]
#include <AlignedMem.h>
#include <ImageMisc.h>
#include <ImageMatrix.h>
#include <MT.h>
#include <MatrixF32.h>

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

namespace fsimd
{
	/*
	* @brief find Min & Max value in an image channel [flat].
	*/
	bool calcMinMaxU8_Iavx(uint8_t* xMin, uint8_t* xMax, const uint8_t* X, size_t size);

	/*
	* @brief caculate mean pixel for an image channel [flat].
	*/
	bool calcMeanU8_Iavx(double* xMean, uint64_t* xSum, const uint8_t* X, size_t size);

	/*
	* @brief convert RGB color image to grayscale image 8bits
	*/
	bool convertRGBtoGrayScale_Iavx2(uint8_t* pImageGS, const RGB32* pImageRGB, size_t size, const float coef[4]);

	/*
	* @brief caculate least square error for linear regression
	*/
	bool calcLeastSquare_Iavx2(double* m, double* b, const double* X, const double* Y, size_t size);

	/*
	* @brief caculate matrix multiplication
	* @param N column size of C == column size of A
	* @param M row size of C == row size of B
	*/
	bool matrixMulF32_Iavx2(float* C, const float* A, const float* B, const int N, const int M, const int S);
}

namespace fwsimd
{
	/*
	* @brief ReLU Layer Aligned
	*/
	bool ReLU_Iavx2(const float* X, float* Y, size_t size);

	/*
	* @brief Leaky ReLU Layer Aligned
	*/
	bool LeakyReLU_Iavx2(const float* X, float* Y, const float negativeSlope, size_t size);

	/*
	* @brief Skip Connection Add Vectors
	*/
	bool Add_Iavx2(const float* X1, const float* X2, float* Y, size_t size);
}
